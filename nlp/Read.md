use export HF_DATASETS_CACHE=“path”

run txtProcessing.py, the paths mentioned in the files is where the text file would be saved.

run the tokenizer.py, it will run and save a snapshot of the model (tokeniser model)

finally run the dataloaderNLP.py. it reads the saved tokenizer ( notices the paths ) and also runs the model for 2 epochs and saves.

The custome dataloader is created: customrNLP
