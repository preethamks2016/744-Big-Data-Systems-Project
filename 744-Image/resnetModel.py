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
    for idx, (imgs,label) in enumerate(data_loader):
        print(idx, (datetime.now() - now).total_seconds() * 1000)
        print(imgs.shape, label.dtype);
        print(imgs.dtype)
        imgs = imgs.float().to(mps_device)
        optims.zero_grad()
        now = datetime.now();
        label=label.cuda()
        output = model(imgs).cuda()
        loss = criterion(output,label)
        loss.backward()
        optims.step()


if __name__ == "__main__":
    main()                                                                                                     
