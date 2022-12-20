import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import h5py
from PIL import Image
import io


class CustomDataset(Dataset):
    listOfTransformations = [lambda img, ksize : cv2.GaussianBlur(img,(5,5),0), lambda img, ksize : cv2.medianBlur(img,5), lambda img, ksize : cv2.blur(img,5)]

    def __init__(self, hf, bs):
        self.hf = hf
        self.batch_size = bs

    def __len__(self):   ##Mandatory override criteria
        return 10000;

    def apply(self,img_to_blur, num):
        for i in range(0,num):
            img_to_blur = self.listOfTransformations[i](img_to_blur,5)
        return img_to_blur

    def __getitem__(self, idx):
        data = self.hf['data'][0:self.batch_size*8]
        inputImages = []
        outputLabels = []
        for new_img in data:
            r1 = random.randint(0, 2)
            img_to_blur = self.apply(new_img, r1)
            img_to_blur = torch.from_numpy(img_to_blur) ##Convert to tensore
            img_to_blur = img_to_blur.permute(2, 0, 1)
            inputImages.append(img_to_blur)
            outputLabels.append(r1)
        
        inputTensor = torch.Tensor(torch.stack([img for img in inputImages]))
        outputTensor = torch.Tensor([r for r in outputLabels])
        return inputTensor , outputTensor
