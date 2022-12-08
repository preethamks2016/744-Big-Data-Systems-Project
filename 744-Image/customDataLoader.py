import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random


class CustomDataset(Dataset):
    listOfTransformations = [lambda img, ksize : cv2.GaussianBlur(img,(5,5),0), lambda img, ksize : cv2.medianBlur(img,5), lambda img, ksize : cv2.blur(img,5)]

    def __init__(self):
        self.imgs_path = "./publaynet/" ##Base path
        file_list = glob.glob(self.imgs_path + "*") ##Contents inside Path
        #train_json = open('./publaynet/train.json')
        self.data = []
        for class_path in file_list:
            for img_path in glob.glob(class_path + "/*.jpg"): ##Loop over all the images
                self.data.append(img_path) ##Check this once we have the data
        print('size', len(self.data))
        #self.class_map = {"dogs" : 0, "cats": 1}
        self.img_dim = (800, 800) ## Reszie dimension experiment

    def __len__(self):   ##Mandatory override criteria
        return len(self.data)

    def apply(self,img_to_blur, num):
        for i in range(0,num):
            img_to_blur = self.listOfTransformations[i](img_to_blur,5)
        return img_to_blur

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img_to_blur = cv2.imread(img_path)
        img_to_blur = cv2.resize(img_to_blur, self.img_dim) ##Should I just return or resize and takecare in transforms
        r1 = random.randint(0, 2)
        img_to_blur = self.apply(img_to_blur, r1)
        img_to_blur = torch.from_numpy(img_to_blur) ##Convert to tensore
        img_to_blur = img_to_blur.permute(2, 0, 1)
        r1 = torch.tensor(random.randint(0, 1))

        return img_to_blur , r1
