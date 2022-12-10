import tokenize
import torch
from datetime import datetime
from customtr import CustomDataset
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import AdamW
from tqdm.auto import tqdm


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
    dataset = CustomDataset()
    print(dataset)
    batch_sizes = 3
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    config = RobertaConfig(
        vocab_size=30_522,  # we align this to the tokenizer vocab_size
        max_position_embeddings=514,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1
    )

    model = RobertaForMaskedLM(config)
    print('model',model)
    model.to(device)

    model.train()
    # initialize optimizer
    optim = AdamW(model.parameters(), lr=1e-4)

    epochs = 10
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        a = datetime.now()
        count = 0;
        for batch in loop:
            print(count, (datetime.now() - a).total_seconds() * 1000)
            # initialize calculated gradients (from prev step)
            # print(len(batch), batch['input_ids'].shape , batch['attention_mask'].shape, batch['labels'].shape)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            optim.zero_grad()
            # pull all tensor batches required for training
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
            count = count + 1
    model.save_pretrained('./vitalian')  # and don't forget to save filiBERTo!

if __name__ == "__main__":
    main()

