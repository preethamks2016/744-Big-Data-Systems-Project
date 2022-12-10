import glob
import cv2
import numpy as np
from PIL import Image
import io
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from minio import Minio


class CustomDataset(Dataset):
    listOfTransformations = [lambda img, ksize : cv2.GaussianBlur(img,(5,5),0), lambda img, ksize : cv2.medianBlur(img,5), lambda img, ksize : cv2.blur(img,5)]

    def __init__(self):
        self.data = []

        client = Minio("localhost:9000",access_key="minioadmin",secret_key="minioadmin", secure=False)
        self.client = client
        objects = client.list_buckets()
        objects = client.list_objects("resnet-images", recursive=True)
        for bucket in objects:
            self.data.append(bucket.object_name)
            # print(bucket.object_name)

        # self.imgs_path = "./publaynet/" ##Base path
        # file_list = glob.glob(self.imgs_path + "*") ##Contents inside Path
        # #train_json = open('./publaynet/train.json')
        
        # for class_path in file_list:
        #     for img_path in glob.glob(class_path + "/*.jpg"): ##Loop over all the images
        #         self.data.append(img_path) ##Check this once we have the data
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
        obj = self.client.get_object(bucket_name = 'resnet-images',object_name= img_path)
        # print(type(obj.data))
        image = np.asarray(bytearray(obj.data), dtype="uint8")
        img_to_blur = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # print(img_to_blur.shape)
        #cv2.imread(img_path)
        img_to_blur = cv2.resize(img_to_blur, self.img_dim) ##Should I just return or resize and takecare in transforms
        r1 = random.randint(0, 2)
        img_to_blur = self.apply(img_to_blur, r1)
        img_to_blur = torch.from_numpy(img_to_blur) ##Convert to tensore
        img_to_blur = img_to_blur.permute(2, 0, 1)
        r1 = torch.tensor(random.randint(0, 1))

        return img_to_blur , r1