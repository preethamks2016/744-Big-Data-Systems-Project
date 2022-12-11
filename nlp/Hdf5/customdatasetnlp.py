import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from transformers import RobertaTokenizer


class CustomDataset(Dataset):    
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('varunItalian', max_len=512)
        self.files_path = "./data/procData/" 
        self.hf = h5py.File(self.files_path, 'r')
        self.mod = 1000
        self.input_ids = None 
        self.attention_mask = None 
        self.labels = None
            
    def __len__(self):   ##Mandatory override criteria
        return len(list(self.hf.keys())) * self.mod ## some random that is big
        
    def __getitem__(self, idx):
        
        dataset_idx = idx // self.mod
        line_idx = idx % self.mod
        if(line_idx == 0):
            self.ds = self.hf[str(dataset_idx)]
            data = np.array(ds[0:self.mod])
            lines = []
            for string in data:
                lines.append(string.decode("utf-8"))
            batch = self.tokenizer(lines, max_length=512, padding='max_length', truncation=True)
            mask = torch.tensor(batch.attention_mask)
            print(mask.shape)
            labels = torch.tensor(batch.input_ids)
            print(labels.shape)
            input_ids = labels.detach().clone()
            rand = torch.rand(input_ids.shape)
            mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
            for i in range(input_ids.shape[0]):
                selection = torch.flatten(mask_arr[i].nonzero()).tolist()
                input_ids[i, selection] = 3  
            self.input_ids = input_ids
            self.attention_mask = mask
            self.labels =  labels         
            return {'input_ids': self.input_ids[0,:], 'attention_mask': self.attention_mask[0,:], 'labels': self.labels[0,:]}      
        else:
            return {'input_ids': self.input_ids[line_idx,:], 'attention_mask': self.attention_mask[line_idx,:], 'labels': self.labels[line_idx,:]}
                        
      

            
            
        
