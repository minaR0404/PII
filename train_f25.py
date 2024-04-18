import json
import copy
import gc
import os
import re
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
import numpy as np
import pandas as pd
from spacy.lang.en import English
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast, DebertaV2Model
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.data.data_collator import DataCollatorForTokenClassification, DefaultDataCollator
from transformers.modeling_outputs import ModelOutput
from peft.mapping import get_peft_config, get_peft_model
from peft.peft_model import PeftModel
from peft import PeftConfig
from peft.tuners.lora import LoraConfig
from peft.utils import TaskType
from datasets import Dataset, DatasetDict, concatenate_datasets

print(torch.cuda.is_available())


TRAINING_MODEL_PATH = "microsoft/deberta-v3-base"
TRAINING_MAX_LENGTH = 2048
EVAL_MAX_LENGTH = 3072
CONF_THRESH = 0.9
LR = 5e-4 #5e-4  # Note: lr for LoRA should be order of magnitude larger than usual fine-tuning
LR_SCHEDULER_TYPE = "linear"
NUM_EPOCHS = 3
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 16 // BATCH_SIZE
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
FREEZE_EMBEDDING = False
FREEZE_LAYERS = 6
LORA_R = 16  # rank of the A and B matricies, the lower the more efficient but more approximate
LORA_ALPHA = LORA_R * 2  # alpha/r is multiplied to BA
AMP = True
N_SPLITS = 4
NEGATIVE_RATIO = 0.3  # downsample ratio of negative samples in the training set
OUTPUT_DIR = "output3" #"output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    fp16=AMP,
    learning_rate=LR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    #gradient_checkpointing=True,  # Gradient Accumulation
    report_to="none",
    evaluation_strategy="steps",
    eval_steps=50,
    eval_delay=50, #100,
    save_strategy="steps",
    save_steps=50, #50,
    save_total_limit=1,
    logging_steps=10,
    label_names=['labels'], #?
    metric_for_best_model="f5", #f3?
    greater_is_better=True,
    load_best_model_at_end=True,
    overwrite_output_dir=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
)


target_modules = ["key_proj", "value_proj", "query_proj", "dense", "classifier"]  # all linear layers
if not FREEZE_EMBEDDING:
    target_modules.append("word_embeddings")  # embedding layer
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, #TaskType.TOKEN_CLS, 
    inference_mode=False,
    target_modules=target_modules,
    modules_to_save=["classifier"],  # The change matrices depends on this randomly initialized matrix, so this needs to be saved
    layers_to_transform=[i for i in range(FREEZE_LAYERS, 24)],
    r=LORA_R,  # The dimension of the low-rank matrices. The higher, the more accurate but less efficient
    lora_alpha=LORA_ALPHA,  # The scaling factor for the low-rank matrices.
    lora_dropout=0.1,
    bias="lora_only"
)


with Path("./data/train.json").open("r") as f:
    original_data = json.load(f)

with Path("./data/mpware_mixtral8x7b_v1.1-no-i-username.json").open("r") as f:
    extra_data = json.load(f)
print("MPWARE's datapoints: ", len(extra_data))


all_labels = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O'
]
id2label = {i: l for i, l in enumerate(all_labels)}
label2id = {v: k for k, v in id2label.items()}
target = [l for l in all_labels if l != "O"]


class CustomTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, label2id: dict, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __call__(self, example: dict) -> dict:
        # rebuild text from tokens
        text, labels, token_map = [], [], []

        for idx, (t, l, ws) in enumerate(
            zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"])
        ):
            text.append(t)
            labels.extend([l] * len(t))
            token_map.extend([idx]*len(t))

            if ws:
                text.append(" ")
                labels.append("O")
                token_map.append(-1)

        text = "".join(text)
        labels = np.array(labels)

        # actual tokenization
        tokenized = self.tokenizer(
            "".join(text),
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length
        )

        token_labels = []

        for start_idx, end_idx in tokenized.offset_mapping:
            # CLS token
            if start_idx == 0 and end_idx == 0:
                token_labels.append(self.label2id["O"])
                continue

            # case when token starts with whitespace
            if text[start_idx].isspace():
                start_idx += 1

            token_labels.append(self.label2id[labels[start_idx]])

        length = len(tokenized.input_ids)

        return {**tokenized, "labels": token_labels, "length": length, "token_map": token_map}
    

tokenizer = DebertaV2TokenizerFast.from_pretrained(TRAINING_MODEL_PATH)
train_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=label2id, max_length=TRAINING_MAX_LENGTH)
eval_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=label2id, max_length=EVAL_MAX_LENGTH)

ds = DatasetDict()

for key, data in zip(["original", "extra"], [original_data, extra_data]):
    ds[key] = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [str(x["document"]) for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "provided_labels": [x["labels"] for x in data],
    })


def find_span(target, document): #(target: list[str], document: list[str]) -> list[list[int]]:
    idx = 0
    spans = []
    span = []

    for i, token in enumerate(document):
        if token != target[idx]:
            idx = 0
            span = []
            continue
        span.append(i)
        idx += 1
        if idx == len(target):
            spans.append(span)
            span = []
            idx = 0
            continue

    return spans


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1+(beta**2))*p*r / ((beta**2)*p + r + 1e-100)
        return fbeta

    def to_dict(self): #-> dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}
    


class MetricsComputer:
    nlp = English()

    def __init__(self, eval_ds: Dataset, label2id: dict, conf_thresh: float = 0.9) -> None:
        self.ds = eval_ds.remove_columns("labels").rename_columns({"provided_labels": "labels"})
        self.gt_df = self.create_gt_df(self.ds)
        self.label2id = label2id
        self.confth = conf_thresh
        self._search_gt()

    def __call__(self, eval_preds: EvalPrediction) -> dict:
        pred_df = self.create_pred_df(eval_preds.predictions)
        return self.compute_metrics_from_df(self.gt_df, pred_df)

    def _search_gt(self) -> None:
        email_regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
        phone_num_regex = re.compile(r"(\(\d{3}\)\d{3}\-\d{4}\w*|\d{3}\.\d{3}\.\d{4})\s")
        self.emails = []
        self.phone_nums = []

        for _data in self.ds:
            # email
            for token_idx, token in enumerate(_data["tokens"]):
                if re.fullmatch(email_regex, token) is not None:
                    self.emails.append(
                        {"document": _data["document"], "token": token_idx, "label": "B-EMAIL", "token_str": token}
                    )
            # phone number
            matches = phone_num_regex.findall(_data["full_text"])
            if not matches:
                continue
            for match in matches:
                target = [t.text for t in self.nlp.tokenizer(match)]
                matched_spans = find_span(target, _data["tokens"])
            for matched_span in matched_spans:
                for intermediate, token_idx in enumerate(matched_span):
                    prefix = "I" if intermediate else "B"
                    self.phone_nums.append(
                        {"document": _data["document"], "token": token_idx, "label": f"{prefix}-PHONE_NUM", "token_str": _data["tokens"][token_idx]}
                    )

    @staticmethod
    def create_gt_df(ds: Dataset):
        gt = []
        for row in ds:
            for token_idx, (token, label) in enumerate(zip(row["tokens"], row["labels"])):
                if label == "O":
                    continue
                gt.append(
                    {"document": row["document"], "token": token_idx, "label": label, "token_str": token}
                )
        gt_df = pd.DataFrame(gt)
        gt_df["row_id"] = gt_df.index

        return gt_df

    def create_pred_df(self, logits: np.ndarray) -> pd.DataFrame:
        """
        Note:
            Thresholing is doen on logits instead of softmax, which could find better models on LB.
        """
        prediction = logits
        o_index = self.label2id["O"]
        preds = prediction.argmax(-1)
        preds_without_o = prediction.copy()
        preds_without_o[:,:,o_index] = 0
        preds_without_o = preds_without_o.argmax(-1)
        o_preds = prediction[:,:,o_index]
        preds_final = np.where(o_preds < self.confth, preds_without_o , preds)

        pairs = set()
        processed = []

        # Iterate over document
        for p_doc, token_map, offsets, tokens, doc in zip(
            preds_final, self.ds["token_map"], self.ds["offset_mapping"], self.ds["tokens"], self.ds["document"]
        ):
            # Iterate over sequence
            for p_token, (start_idx, end_idx) in zip(p_doc, offsets):
                label_pred = id2label[p_token]

                if start_idx + end_idx == 0:
                    # [CLS] token i.e. BOS
                    continue

                if token_map[start_idx] == -1:
                    start_idx += 1

                # ignore "\n\n"
                while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                    start_idx += 1

                if start_idx >= len(token_map):
                    break

                token_id = token_map[start_idx]
                pair = (doc, token_id)

                # ignore "O", preds, phone number and  email
                if label_pred in ("O", "B-EMAIL", "B-PHONE_NUM", "I-PHONE_NUM") or token_id == -1:
                    continue

                if pair in pairs:
                    continue

                processed.append(
                    {"document": doc, "token": token_id, "label": label_pred, "token_str": tokens[token_id]}
                )
                pairs.add(pair)

        pred_df = pd.DataFrame(processed + self.emails + self.phone_nums)
        pred_df["row_id"] = list(range(len(pred_df)))

        return pred_df

    def compute_metrics_from_df(self, gt_df, pred_df):
        """
        Compute the LB metric (lb) and other auxiliary metrics
        """

        references = {(row.document, row.token, row.label) for row in gt_df.itertuples()}
        predictions = {(row.document, row.token, row.label) for row in pred_df.itertuples()}

        score_per_type = defaultdict(PRFScore)
        references = set(references)

        for ex in predictions:
            pred_type = ex[-1] # (document, token, label)
            if pred_type != 'O':
                pred_type = pred_type[2:] # avoid B- and I- prefix

            if pred_type not in score_per_type:
                score_per_type[pred_type] = PRFScore()

            if ex in references:
                score_per_type[pred_type].tp += 1
                references.remove(ex)
            else:
                score_per_type[pred_type].fp += 1

        for doc, tok, ref_type in references:
            if ref_type != 'O':
                ref_type = ref_type[2:] # avoid B- and I- prefix

            if ref_type not in score_per_type:
                score_per_type[ref_type] = PRFScore()
            score_per_type[ref_type].fn += 1

        totals = PRFScore()

        for prf in score_per_type.values():
            totals += prf

        return {
            "precision": totals.precision,
            "recall": totals.recall,
            "f5": totals.f5,
            **{
                f"{v_k}-{k}": v_v
                for k in set([l[2:] for l in self.label2id.keys() if l!= 'O'])
                for v_k, v_v in score_per_type[k].to_dict().items()
            },
        }


class F25Collator():
    def __init__(self, tokenizer, max_length=TRAINING_MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        text, lbl = [], []
        for idx, (t, l, ws) in enumerate(zip(examples["tokens"], examples["provided_labels"], examples["trailing_whitespace"])):
            text.append(t)
            lbl.extend([l] * len(t))
            if ws:
                text.append(" ")
                lbl.append("O")

        text = "".join(text)
        lbl = np.array(lbl)

        tokenized = self.tokenizer(
            "".join(text),
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length
        )

        token_labels = []
        for start_idx, end_idx in tokenized.offset_mapping:
            # CLS token
            if start_idx == 0 and end_idx == 0:
                token_labels.append(self.label2id["O"])
                continue
            # case when token starts with whitespace
            if text[start_idx].isspace():
                start_idx += 1
            token_labels.append(self.label2id[lbl[start_idx]])
        tokenized['labels'] = torch.tensor(token_labels)
        #encodings['category_id'] = torch.tensor(examples['category_id'])
        return tokenized

#https://qiita.com/m__k/items/2c4e476d7ac81a3a44af#1-%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E6%88%BB%E3%82%8A%E5%80%A4%E3%82%92modeloutput%E3%81%AB%E3%81%99%E3%82%8B
class F25Net(nn.Module):
    def __init__(self, model, num_labels, loss_function=None):
        super().__init__()
        self.model = model
        self.hidden_size = self.model.config.hidden_size
        self.linear = nn.Linear(self.hidden_size, num_labels)
        self.loss_function = loss_function

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds =None,
                output_attentions=False,
                output_hidden_states=False,
                labels=None):
        
        outputs = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states)
        
        state = outputs.last_hidden_state[:, 0, :]
        state = self.linear(state)
        print(state.shape)
        print(labels.shape)

        loss=None
        if labels is not None and self.loss_function is not None:
            loss = self.loss_function(state, labels)

        attentions=None
        if output_attentions:
            attentions=outputs.attentions
        
        hidden_states=None
        if output_hidden_states:
            hidden_states=outputs.hidden_states

        return ModelOutput(
            logits=state,
            loss=loss,
            last_hidden_state=outputs.last_hidden_state,
            attentions=attentions,
            hidden_states=hidden_states
        )


f25_collator = F25Collator(tokenizer)

loss_fct = nn.CrossEntropyLoss()
model = DebertaV2Model.from_pretrained(TRAINING_MODEL_PATH)
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
net = F25Net(model, len(label2id), loss_fct)


# split according to document id
folds = [
    (
        np.array([i for i, d in enumerate(ds["original"]["document"]) if int(d) % N_SPLITS != s]),
        np.array([i for i, d in enumerate(ds["original"]["document"]) if int(d) % N_SPLITS == s])
    )
    for s in range(N_SPLITS)
]

negative_idxs = [i for i, labels in enumerate(ds["original"]["provided_labels"]) if not any(np.array(labels) != "O")]
exclude_indices = negative_idxs[int(len(negative_idxs) * NEGATIVE_RATIO):]


for fold_idx, (train_idx, eval_idx) in enumerate(folds):
    args.run_name = f"fold-{fold_idx}"
    args.output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    original_ds = ds["original"].select([i for i in train_idx if i not in exclude_indices])
    train_ds = concatenate_datasets([original_ds, ds["extra"]])
    train_ds = train_ds.map(train_encoder, num_proc=1) #os.cpu_count()
    eval_ds = ds["original"].select(eval_idx)
    eval_ds = eval_ds.map(eval_encoder, num_proc=1)
    print(train_ds)
    trainer = Trainer(
        args=args,
        model=net,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=MetricsComputer(eval_ds=eval_ds, label2id=label2id),
        #data_collator=f25_collator,
    )
    trainer.train()
    eval_res = trainer.evaluate(eval_dataset=eval_ds)
    with open(os.path.join(args.output_dir, "eval_result.json"), "w") as f:
        json.dump(eval_res, f)
    # merge LoRA adapters back to the original weights
    trainer.model = trainer.model.base_model.merge_and_unload()
    trainer.save_model(os.path.join(OUTPUT_DIR, f"fold_{fold_idx}", "best"))
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    break  # delete this line to perform 4-fold cross-validation
print("end.")
