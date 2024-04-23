import json
import os
import gc
import re
import bisect
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch import nn
from transformers.modeling_outputs import ModelOutput
from datasets import Dataset
from spacy.lang.en import English
from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast, DebertaV2Model
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForTokenClassification


INFERENCE_MAX_LENGTH = 3500
CONF_THRESH = 0.90  # threshold for "O" class
URL_THRESH = 0.1  # threshold for URL
V2_THRESH = 0.95  # threshold for v2
PRED_THRESH = 0.05  # threshold for precision
AMP = True
MODEL_PATH = {"./data/output3/fold_0/best/": 1.00,
}
DATA_DIR = './data/'


nlp = English()

def find_span(target: list[str], document: list[str]) -> list[list[int]]:
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

def spacy_to_hf(data: dict, idx: int) -> slice:
    """
    Given an index of spacy token, return corresponding indices in deberta's output.
    We use this to find indice of URL tokens later.
    """
    str_range = np.where(np.array(data["token_map"]) == idx)[0]
    start_idx = bisect.bisect_left([off[1] for off in data["offset_mapping"]], str_range.min())
    end_idx = start_idx
    while end_idx < len(data["offset_mapping"]):
        if str_range.max() > data["offset_mapping"][end_idx][1]:
            end_idx += 1
            continue
        break
    token_range = slice(start_idx, end_idx+1)
    return token_range

class CustomTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, example: dict) -> dict:
        text = []
        token_map = []

        for idx, (t, ws) in enumerate(zip(example["tokens"], example["trailing_whitespace"])):
            text.append(t)
            token_map.extend([idx]*len(t))
            if ws:
                text.append(" ")
                token_map.append(-1)

        tokenized = self.tokenizer(
            "".join(text),
            return_offsets_mapping=True,
            truncation=False, #True,
            #max_length=self.max_length,
        )

        return {**tokenized,"token_map": token_map,}
    

# with open(str(Path(DATA_DIR).joinpath("test.json")), "r") as f:
#     data = json.load(f)


## if CV,
with open(str(Path(DATA_DIR).joinpath("train.json")), "r") as f:
    cvdata = json.load(f)
    
data = []
for dat in cvdata:
    if dat["document"]%4 == 0:
        data.append({
            "document": dat["document"],
            "full_text": dat["full_text"],
            "tokens": dat["tokens"],
            "trailing_whitespace": dat["trailing_whitespace"],
            "labels": dat["labels"],
        })


first_model_path = list(MODEL_PATH.keys())[0]

id2label = { 0: 'B-EMAIL',
             1: 'B-ID_NUM',
             2: 'B-NAME_STUDENT',
             3: 'B-PHONE_NUM',
             4: 'B-STREET_ADDRESS',
             5: 'B-URL_PERSONAL',
             6: 'B-USERNAME',
             7: 'I-ID_NUM',
             8: 'I-NAME_STUDENT',
             9: 'I-PHONE_NUM',
             10: 'I-STREET_ADDRESS',
             11: 'I-URL_PERSONAL',
             12: 'O'}
label2id = { 'B-EMAIL': 0,
             'B-ID_NUM': 1,
             'B-NAME_STUDENT': 2,
             'B-PHONE_NUM': 3,
             'B-STREET_ADDRESS': 4,
             'B-URL_PERSONAL': 5,
             'B-USERNAME': 6,
             'I-ID_NUM': 7,
             'I-NAME_STUDENT': 8,
             'I-PHONE_NUM': 9,
             'I-STREET_ADDRESS': 10,
             'I-URL_PERSONAL': 11,
             'O': 12}


total_weight = sum(MODEL_PATH.values())


def get_final(MODEL_PATH, weighted_average_predictions):
    #first_model_path = list(MODEL_PATH.keys())[0]
    #model = DebertaV2ForTokenClassification.from_pretrained(first_model_path)
    #id2label = model.config.id2label
    o_index = label2id["O"]
    preds = weighted_average_predictions.argmax(-1)
    preds_without_o = weighted_average_predictions.copy()
    preds_without_o[:,:,o_index] = 0
    preds_without_o = preds_without_o.argmax(-1)
    o_preds = weighted_average_predictions[:,:,o_index]
    preds_final = np.where(o_preds < CONF_THRESH, preds_without_o , preds)
    
    return preds_final


def get_processed(preds_final, weighted_average_predictions, ds):

    processed =[]
    pairs = set()

    # Iterate over document
    for p, ap, token_map, offsets, tokens, doc in zip(
        preds_final, weighted_average_predictions, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"]
    ):
        # Iterate over sequence
        for token_pred, apreds, (start_idx, end_idx) in zip(p, ap, offsets):
            label_pred = id2label[token_pred]

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

            # ignore certain labels and whitespace
            if label_pred in ("O", "B-EMAIL", "B-URL_PERSONAL", "B-PHONE_NUM", "I-PHONE_NUM") or token_id == -1:
                continue        

            if pair in pairs:
                continue
                
            processed.append(
                {"document": doc, "token": token_id, "label": label_pred, "token_str": tokens[token_id], "preds": apreds[label2id[label_pred]], "O-preds": apreds[12]}
            )
            pairs.add(pair)
            
    return processed, pairs


def get_url(processed, weighted_average_predictions, ds, pairs):

    url_whitelist = [
        "wikipedia.org",
        "coursera.org",
        "google.com",
        ".gov",
    ]
    url_whitelist_regex = re.compile("|".join(url_whitelist))

    for row_idx, _data in enumerate(ds):
        for token_idx, token in enumerate(_data["tokens"]):
            if not nlp.tokenizer.url_match(token):
                continue
            print(f"Found URL: {token}")
            if url_whitelist_regex.search(token) is not None:
                print("The above is in the whitelist")
                continue
            input_idxs = spacy_to_hf(_data, token_idx)
            probs = weighted_average_predictions[row_idx, input_idxs, label2id["B-URL_PERSONAL"]]
            if probs.mean() > URL_THRESH:
                print("The above is PII")
                processed.append(
                    {
                        "document": _data["document"], 
                        "token": token_idx, 
                        "label": "B-URL_PERSONAL", 
                        "token_str": token,
                        "preds": probs.mean(),
                        "O-preds": weighted_average_predictions[row_idx, input_idxs, 12].mean()
                    }
                )
                pairs.add((_data["document"], token_idx))
            else:
                print("The above is not PII")
                
    return processed


def get_email_phone(ds):

    email_regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
    phone_num_regex = re.compile(r"(\(\d{3}\)\d{3}\-\d{4}\w*|\d{3}\.\d{3}\.\d{4})\s")
    emails = []
    phone_nums = []

    for _data in ds:
        # email
        for token_idx, token in enumerate(_data["tokens"]):
            if re.fullmatch(email_regex, token) is not None:
                emails.append(
                    {"document": _data["document"], "token": token_idx, "label": "B-EMAIL", "token_str": token}
                )
        # phone number
        matches = phone_num_regex.findall(_data["full_text"])
        if not matches:
            continue
        for match in matches:
            target = [t.text for t in nlp.tokenizer(match)]
            matched_spans = find_span(target, _data["tokens"])
        for matched_span in matched_spans:
            for intermediate, token_idx in enumerate(matched_span):
                prefix = "I" if intermediate else "B"
                phone_nums.append(
                    {"document": _data["document"], "token": token_idx, "label": f"{prefix}-PHONE_NUM", "token_str": _data["tokens"][token_idx]}
                )
                
    return emails, phone_nums


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
        
        state = outputs.last_hidden_state[:, :, :]
        state = self.linear(state)
        state = state.squeeze()
        labels = labels.squeeze()
        
        # label13 = torch.zeros((labels.shape[1], 13), device="cuda:0")
        # for l in range(13):
        #     label13[:,l] = labels[:,:]

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


for idx, (model_path, weight) in enumerate(MODEL_PATH.items()):
    print(model_path)
    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [x["document"] for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    })

    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_path)
    ds = ds.map(CustomTokenizer(tokenizer=tokenizer, max_length=INFERENCE_MAX_LENGTH), num_proc=os.cpu_count())

    model = DebertaV2Model.from_pretrained(model_path)
    #collator = DataCollatorForTokenClassification(tokenizer)
    args = TrainingArguments(".", per_device_eval_batch_size=1, report_to="none", fp16=AMP)
    trainer = Trainer(
        model=model, args=args, tokenizer=tokenizer,
    )

    predictions = trainer.predict(ds).predictions  # (n_sample, len, n_labels)

    weighted_average_predictions = torch.softmax(torch.from_numpy(predictions), dim=2).numpy() * weight
    # Save weighted_predictions to disk
    #np.save(os.path.join(intermediate_dir, f'weighted_preds_{idx}.npy'), weighted_predictions)

    # Clear memory
    del ds, model, args, trainer, tokenizer, predictions #, weighted_predictions
    torch.cuda.empty_cache()
    gc.collect()


ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [x["document"] for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    })

tokenizer = DebertaV2TokenizerFast.from_pretrained('/kaggle/input/37vp4pjt')
ds = ds.map(CustomTokenizer(tokenizer=tokenizer, max_length=INFERENCE_MAX_LENGTH), num_proc=os.cpu_count())


preds_final = get_final(MODEL_PATH, weighted_average_predictions)

processed, pairs = get_processed(preds_final, weighted_average_predictions, ds)
processed = get_url(processed, weighted_average_predictions, ds, pairs)
emails, phone_nums = get_email_phone(ds)

df = pd.DataFrame(processed + emails + phone_nums)
df.head(100)


