import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from transformers import RobertaTokenizer


class CustomDataset(Dataset):    
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('./varunItalian', max_len=512)
        self.files_path = "/home/rkosgi/744Project/nlp/data/text/oscar_it/" 
        file_list = glob.glob(self.files_path + "*") ##Contents inside Path
        self.fileNames = []
        for class_path in file_list:
            for filepath in glob.glob(class_path): ##Loop over all the images
                print(filepath)
                self.fileNames.append(filepath) ##Check this once we have the data
        print('Number of files', len(self.fileNames))
        self.dataArr = []
        self.mod = 1000
        self.fileOpen = None
        self.input_ids = None
        self.attention_mask = None
        self.labels = None
            
    def __len__(self):   ##Mandatory override criteria
        return len(self.fileNames) * self.mod ## some random that is big
        
    def __getitem__(self, idx):
        
        #Add some randome file openings ask !!!!
        #possible randomsize on files, batch sizes and encoding ,1 = 32
        print(idx, "HW", len(self.fileNames) * self.mod)
        file_idx = idx // self.mod;
        line_idx = idx % self.mod
        print('HI', line_idx)
        file_path = self.fileNames[file_idx]
        if(line_idx == 0):
            with open(file_path, 'r', encoding='utf-8') as fp:
                lines = fp.read().split('\n')
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
        
            
            
       
        
        
        # with open(file_path, 'r', encoding='utf-8') as fp:
        #     lines = fp.read().split('\n')
        # lineArr = lines[:8];
        # print("lines:",len(lineArr))
        # batch = self.tokenizer(lineArr, max_length=512, padding='max_length', truncation=True)
        # #print(len(batch.input_ids))
        # mask = torch.tensor(batch.attention_mask)
        # labels = torch.tensor(batch.input_ids)
        # input_ids = labels.detach().clone()
        # rand = torch.rand(input_ids.shape)
        # mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
        # for i in range(input_ids.shape[0]):
        #     selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        #     input_ids[i, selection] = 3  # our custom [MASK] token == 3
    
        # self.encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
        #print(input_ids.shape,mask.shape,labels.shape,"h")
        

            
            
        
