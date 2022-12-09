import glob
from customDataLoader import CustomDataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AdamW
import torch.optim as optim
import torch.nn as nn

mps_device = torch.device("cuda:0")
print(mps_device)
def main():
    dataset = CustomDataset()
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False).cuda();
    optims = optim.SGD(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    now = datetime.now()
    timeiterStart = datetime.now()
    iterTime = []
    dataLoadTime = []
    for idx, (imgs,label) in enumerate(data_loader):
        loadtime = (datetime.now() - now).total_seconds() * 1000

        print( "iteration - ", idx, " dataloading time ",loadtime)
        # print(imgs.shape, label.dtype);
        # print(imgs.dtype)
        imgs = imgs.float().to(mps_device)
        optims.zero_grad()
        label=label.cuda()
        output = model(imgs).cuda()
        loss = criterion(output,label)
        loss.backward()
        optims.step()
        now = datetime.now()
        timeiterEnd= datetime.now()
        iterationTime = (timeiterEnd-timeiterStart).total_seconds() * 1000
        print("iteration - ", idx, "iteration time"  ,iterationTime)
        timeiterStart = datetime.now()
        iterTime.append(iterationTime)
        dataLoadTime.append(loadtime)
        if(idx%20 == 0 and idx!=0):
            print("average iteration time - ", sum(iterTime)/len(iterTime))
            print("average load time - ", sum(dataLoadTime)/len(dataLoadTime))
            dataLoadTime = []
            iterTime = []





if __name__ == "__main__":
    main()                                                                                                     
