import glob
from customDataLoader import CustomDataset
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AdamW
import torch.optim as optim
import torch.nn as nn
import csv

mps_device = torch.device("cuda:0")
print(mps_device)
def main(batchSize, numWorkers, outputFile, numThreads):
    if(numThreads>0):
        torch.set_num_threads(numThreads)
    dataset = CustomDataset()
    data_loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False).cuda();
    optims = optim.SGD(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    now = datetime.now()
    timeiterStart = datetime.now()
    iterTime = []
    dataLoadTime = []
    fields = ['LoadTime', 'IterationTime']
    metrics = []

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
            metrics.append([sum(dataLoadTime)/len(dataLoadTime), sum(iterTime)/len(iterTime)])
            dataLoadTime = []
            iterTime = []
        if(idx==80):
            break
    with open(outputFile, 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(metrics)





if __name__ == "__main__":
    parser = argparse.ArgumentParser("PyTorch - Training ResNet101 on CIFAR10 Dataset")
    # parser.add_argument('--num_nodes', type=int, default=1, help='total number of processes')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--output_file', type=str, default='metrics_ext4.csv', help='metrics file name')
    parser.add_argument('--num_threads', type=int, default=0, help='number of threads')

    
    args = parser.parse_args()
    # print(args)
    batchSize = 64
    port =  "23456"
    main(args.batch_size,args.num_workers, args.output_file,args.num_threads )
    # main()                                                                                                     
