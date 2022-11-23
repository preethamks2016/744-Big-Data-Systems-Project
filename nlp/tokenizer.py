from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

import os


paths = [str(x) for x in Path('./data/text/oscar_it').glob('**/*.txt')]
print(paths)
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=paths[:5], vocab_size=30522, min_frequency=2,special_tokens=['<s>', '</s>', '<unk>', '<pad>', '<mask>'])
os.mkdir('./varunItalian')
tokenizer.save_model('varunItalian')