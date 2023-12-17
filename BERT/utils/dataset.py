from numpy import char, character
from numpy.core.defchararray import encode
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import csv
import re


'''
read the data from the text(csv) file
and extract the (text, label) pairs
'''

class HotelDataset(Dataset):
    def __init__(self, data_path, max_length):
        df = pd.read_csv(data_path, delimiter='\t', header=None, names=['label', 'sentence'])

        sentences = df.sentence.values
        labels = df.label.values


        #  filter out the characters not recognized and english text.
        pattern_characters = r'&#\d{5};'
        pattern_letters = r'[a-zA-Z]'
        sentences = [re.sub(pattern_characters, '', sentence) for sentence in sentences]
        sentences = [re.sub(pattern_letters, '', sentence) for sentence in sentences]
        

        #  tokenize the text and pad and truncate
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        encoding = tokenizer(sentences, padding = 'max_length', truncation=True, max_length=max_length)
        
        self.encoded_sentences = encoding['input_ids']
        self.attention_masks   = encoding['attention_mask']
        self.labels            = labels


    def __len__(self):
        return len(self.encoded_sentences)


    def __getitem__(self, idx):
        encoded_sentence = self.encoded_sentences[idx]
        attention_mask   = self.attention_masks[idx]
        label            = self.labels[idx]

        #  tensorlize
        sentence_tensor = torch.tensor(encoded_sentence , dtype = torch.int32)
        mask_tensor     = torch.tensor(attention_mask   , dtype = torch.int32)
        label_tensor    = torch.tensor(label            , dtype = torch.long)

        
        return sentence_tensor, mask_tensor, label_tensor
