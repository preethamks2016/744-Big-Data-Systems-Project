from datasets import load_dataset
from tqdm.auto import tqdm

dataset = load_dataset('oscar', 'unshuffled_deduplicated_it')
print(dataset)

print(dataset['train'])

print(dataset['train'].features)

print(dataset['train'][0])

text_data = []
file_count = 0

for sample in tqdm(dataset['train']):
    sample = sample['text'].replace('\n', '')
    text_data.append(sample)
    if len(text_data) == 10_000:
        # once we git the 10K mark, save to file
        with open(f'./data/text/oscar_it/text_{file_count}.txt', 'w+', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1
# after saving in 10K chunks, we will have ~2082 leftover samples, we save those now too
with open(f'/data/text/oscar_it/text_{file_count}.txt', 'w+', encoding='utf-8') as fp:
    fp.write('\n'.join(text_data))
