import tokenize
from transformers import RobertaTokenizer
import torch
from customItalianDataset import CustomItalianDataset
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import AdamW
from tqdm.auto import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
##############
##FOR ONE FILE :  preprop
tokenizer = RobertaTokenizer.from_pretrained('varunItalian', max_len=512)


with open('./data/text/oscar_it/text_1.txt', 'r', encoding='utf-8') as fp:
    lines = fp.read().split('\n')
print(len(lines))
batch = tokenizer(lines, max_length=512, padding='max_length', truncation=True)
mask = torch.tensor(batch.attention_mask)
print(mask.dtype)
labels = torch.tensor(batch.input_ids)
print(labels.dtype)

input_ids = labels.detach().clone()
rand = torch.rand(input_ids.shape)
mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)

for i in range(input_ids.shape[0]):
    # get indices of mask positions from mask array
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    # mask input_ids
    input_ids[i, selection] = 3  # our custom [MASK] token == 3
    
encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}


#####Clarity
dataset = CustomItalianDataset(encodings)

loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)


config = RobertaConfig(
    vocab_size=30_522,  # we align this to the tokenizer vocab_size
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

model = RobertaForMaskedLM(config)
model.to(device)

model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=1e-4)

epochs = 2
for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
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
        
model.save_pretrained('./vitalian')  # and don't forget to save filiBERTo!