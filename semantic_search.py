from json import load
import os
import streamlit as st
import pandas as pd
import torch
import transformers
import pandas as pd
from sentence_transformers import models, SentenceTransformer
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional
from torch.autograd import Variable
from sklearn.decomposition import PCA
from rich import print
#from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
plt.show()
from pytorch_lightning.loggers import TensorBoardLogger
import gc
gc.collect()
from sklearn.model_selection import train_test_split
torch.cuda.empty_cache()
import os.path as osp
from typing import Any, Dict, List, Optional, Type
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from sklearn.model_selection import KFold
from transformers import BertModel, BertConfig
from transformers import DistilBertModel, DistilBertConfig , DistilBertTokenizerFast
from dotenv import load_dotenv
load_dotenv()

import pinecone

BERT_MODEL_NAME = 'distilbert-base-uncased'
# 37 label
def greet(name):
    return "Hello " + name + "!!"
def predict(text):
    print(5)
if __name__ == '__main__':
    API_KEY = os.getenv("PINECONE_API_KEY")
    print(API_KEY)

    pinecone.init(api_key = API_KEY , environment='us-west1-gcp')
    bert = models.Transformer('nreimers/albert-small-v2')
    bert
    pooler = models.Pooling(
        bert.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    model = SentenceTransformer(modules=[bert, pooler])    ## load pytorch model weight from retriever.pth
    model.load_state_dict(torch.load('retriever.pth' , map_location='cpu'))
    print(model)
    query= "What are hooks in pytorch lightning?"
    index = pinecone.Index('github-question-answer')
    xq = model.encode([query]).tolist()
    xc = index.query(xq , top_k = 3 , include_metadata = True)
    print(xc)
    print(xc["result"])