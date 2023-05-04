import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

class DepressDataset(Dataset):
    def __init__(self, file_path, mode):
        super().__init__()
        self.mode = mode
        df = pd.read_csv(file_path, sep='\t')
        dic = {'not depression': 0, 'moderate': 1, 'severe': 2}
        if mode != 'test':
            df['Label'] = df['Label'].map(dic)
            self.labels = df['Label'].tolist()
        # ugly expident, sorry but it works
        if mode == "test":
            df['Class labels'] = df['Class labels'].map(dic)
            self.labels = df['Class labels'].tolist()
        self.data = {}
        for idx, row in df.iterrows():
            if mode != 'test':
                if mode == 'train':
                    self.data[idx] = (row['Text_data'], row['neg'], row['neu'], row['pos'], row['compound'], row['Label'])
                elif mode == 'dev':
                    self.data[idx] = (row['Text data'], row['neg'], row['neu'], row['pos'], row['compound'], row['Label'])
            else:
                self.data[idx] = (row['text data'], row['neg'], row['neu'], row['pos'], row['compound'],row['Class labels'])
                # self.data[idx] = (row['Text data'], row['neg'], row['neu'], row['pos'], row['compound'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode != 'test':
            text, neg, neu, pos, compound, label = self.data[idx]
            vad_score = [neg, neu, pos, compound]
            return (text, torch.tensor(vad_score), torch.tensor(label, dtype=torch.long))
        else: # 是一样的，但我们项目中只用到这一次，就这样写了
            # text, neg, neu, pos, compound = self.data[idx]
            # vad_score = [neg, neu, pos, compound]
            # return (text, torch.tensor(vad_score))
            text, neg, neu, pos, compound, label = self.data[idx]
            vad_score = [neg, neu, pos, compound]
            return (text, torch.tensor(vad_score), torch.tensor(label, dtype=torch.long))

    def get_labels(self):
        return self.labels