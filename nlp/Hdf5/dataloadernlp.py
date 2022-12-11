import tokenize
import torch
from datetime import datetime
from customdatasetnlp import CustomDataset
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import AdamW
from tqdm.auto import tqdm
import argparse
import csv

print("before", "hii")
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()
def main(batchSize, numWorkers, outputFile, numThreads):
    
    if(numThreads>0):
        torch.set_num_threads(numThreads)

    dataset = CustomDataset()
    # print(dataset)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)

    config = RobertaConfig(
        vocab_size=30_522,  # we align this to the tokenizer vocab_size
        max_position_embeddings=514,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1
    )

    model = RobertaForMaskedLM(config)
    # print('model',model)
    model.to(device)

    model.train()
    # initialize optimizer
    optim = AdamW(model.parameters(), lr=1e-4)

    epochs = 1
    now = datetime.now()
    timeiterStart = datetime.now()
    iterTime = []
    dataLoadTime = []
    metrics = []
    fields = ['LoadTime', 'IterationTime']
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        a = datetime.now()
        count = 0;
        for batch in loop:
            loadtime = (datetime.now() - now).total_seconds() * 1000
            print(count)
            # initialize calculated gradients (from prev step)
            # print(len(batch), batch['input_ids'].shape , batch['attention_mask'].shape, batch['labels'].shape)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            optim.zero_grad()
            # pull all tensor batches required for traininçgx
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            a = datetime.now();
            now = datetime.now()
            timeiterEnd= datetime.now()
            iterationTime = (timeiterEnd-timeiterStart).total_seconds() * 1000
            timeiterStart = datetime.now()
            iterTime.append(iterationTime)
            dataLoadTime.append(loadtime)
            if(count%20 == 0 and count!=0):
                print("average iteration time - ", sum(iterTime)/len(iterTime))
                print("average load time - ", sum(dataLoadTime)/len(dataLoadTime))
                metrics.append([sum(dataLoadTime)/len(dataLoadTime), sum(iterTime)/len(iterTime)])
                dataLoadTime = []
                iterTime = []
            if(count==40):
                break
            count = count + 1
    model.save_pretrained('./vitalian')  # and don't forget to save filiBERTo!
    
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
    main(args.batch_size,args.num_workers, args.output_file,args.num_threads)

