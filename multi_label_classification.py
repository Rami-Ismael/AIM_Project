import streamlit as st
import pickle
import pandas as pd
import torch
import transformers
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from rich import print
#from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
torch.cuda.empty_cache()
import numpy as onp
import os.path as osp
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
from pytorch_lightning.loops.base import Loop
from rich.pretty import pprint
#import bitsandbytes as bnb
from transformers import DistilBertModel, DistilBertConfig , DistilBertTokenizerFast
BERT_MODEL_NAME = 'distilbert-base-uncased'
MAX_TOKEN_LEN = 2<<8
index_name = {
    "1297090688": 'feature',
    "1297090689": 'help wanted',
    "1297090686": 'bug',
    "1297090692": 'question',
    "1297090690": 'good first issue',
    "2253758944": 'logger',
    "1297090693": "won't fix",
    "1475966033": 'example',
    "1840917107": 'docs',
    "1851720487": 'priority: 0',
    "2251044906": 'strategy: dp',
    "1893143017": "let's do it!",
    "1862633788": 'discussion',
    "2014560835": 'waiting on author',
    "1297090687": 'duplicate',
    "2245077676": 'design',
    "1819743298": 'ready',
    "1851722664": 'ci',
    "1973825445": 'accelerator: tpu',
    "2122110652": 'working as intended',
    "2604053872": 'enhancement',
    "2604053875": 'help wanted',
    "2651599587": 'has conflicts',
    "2368743110": '3rd party',
    "2477304581": 'priority: 2',
    "1934189816": 'enhancement',
    "1934189817": 'good first issue',
    "1934189818": 'help wanted',
    "1934189821": "won't fix",
    "2615223038": 'logger',
    "2244367919": 'data handling',
    "2823174111": 'New metric',
    "3240345158": 'wontfix',
    "2324728524": 'checkpointing',
    "2240244232": 'refactor',
    "2237125337": 'distributed',
    "2604053874": 'good first issue',
    "2477301440": 'priority: 1'
}
id_index = {
    "1840917107": 0,
    "2253758944": 1,
    "1297090692": 2,
    "1297090688": 3,
    "1297090689": 4,
    "1297090693": 5,
    "1297090686": 6,
    "1819743298": 7,
    "1851720487": 8,
    "1851722664": 9,
    "1862633788": 10,
    "1973825445": 11,
    "1297090687": 12,
    "1893143017": 13,
    "1297090690": 14,
    "2014560835": 15,
    "2324728524": 16,
    "1475966033": 17,
    "2237125337": 18,
    "2251044906": 19,
    "2477301440": 20,
    "2245077676": 21,
    "2604053872": 22,
    "2604053874": 23,
    "2604053875": 24,
    "2823174111": 25,
    "2368743110": 26,
    "2477304581": 27,
    "2122110652": 28,
    "2244367919": 29,
    "2240244232": 30,
    "3240345158": 31,
    "1934189816": 32,
    "1934189817": 33,
    "1934189818": 34,
    "1934189821": 35,
    "2615223038": 36,
    "2651599587": "37"
}
# 37 label
def greet(name):
    return "Hello " + name + "!!"
def predict(text):
    print(5)
class OPClassifier(pl.LightningModule):
  def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None, learning_rate = 2e-5):
    super().__init__()
    self.bert = DistilBertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.learning_rate = learning_rate
    ##https://discuss.pytorch.org/t/bceloss-are-unsafe-to-autocast/110407
    self.criterion = nn.BCEWithLogitsLoss()
  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids = input_ids, attention_mask=attention_mask , output_hidden_states = True)
    output = self.classifier(output.last_hidden_state[:, 0])
    ##https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
    #output = torch.sigmoid(output)
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output
  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    labels = labels.detach().cpu().type(torch.IntTensor)
    outputs = outputs.detach().cpu()
    return {"loss": loss}
  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    labels = labels.detach().cpu().type(torch.IntTensor)
    outputs = outputs.detach().cpu()
    return loss
  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    labels = labels.detach().cpu().type(torch.IntTensor)
    outputs = outputs.detach().cpu()
    return loss
  '''
  def training_epoch_end(self, outputs):
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)
    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)
    #for i, name in enumerate(LABEL_COLUMNS):
      #class_roc_auc = auroc(predictions[:, i], labels[:, i])
      #self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
  '''
  def configure_optimizers(self):
    optimizer = torch.optim.AdamW8bit(self.bert.parameters(), lr=self.learning_rate, betas=(0.9, 0.995)) # add bnb optimizer
    scheduler = transformers.get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )
    #return optimizer
    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )
if __name__ == '__main__':
    '''
    image = gr.inputs.Image(shape=(224, 224))
    label = gr.outputs.Label(num_top_classes=3)
    iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    iface.launch(share= True)
    '''
    df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })
    ## add a text area for User input text
    # st.write('Sentiment:', run_sentiment_analysis(txt))
    query = "What are hooks in pytorch lightning?"
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    title = st.text_input(label = 'Insert Text', placeholder = 'Insert query here')
    encoding = tokenizer.encode_plus(
      query,
      add_special_tokens=True,
      max_length=MAX_TOKEN_LEN,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    ## load pytorch model weight from retriever.pth
    model = OPClassifier(
    n_classes=38,
    n_warmup_steps=1,
    n_training_steps=1
    )
    model.load_state_dict(torch.load("/home/ramiismael/code/machine_learning_web_app_on_azure/classifier.pth"))
    model.eval()
    print(model(**encoding))
    print(model(**encoding)[1].shape)
    last_layer  = model(**encoding)[1]
    answer = torch.sigmoid(last_layer)
    print(answer)
    answer = answer.detach().cpu().numpy()
    answer = answer.tolist()
    print(answer)

    for idx , x  in enumerate(answer[0]):
      if x>.50:
        for key , value in id_index.items():
          if value == idx:
            st.headers(index_name[key])