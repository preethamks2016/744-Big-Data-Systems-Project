import h5py
import numpy as np
import os
import io


save_path = './text1.hdf5'
hf = h5py.File(save_path, 'w') # open a hdf5 file
folder_dir = "/home/rkosgi/744Project/nlp/data/text/oscar_it/"

idx = 0
spamFiles = []
for textFile in os.listdir(folder_dir):
    if (textFile.endswith(".txt")):
        file1 = open(os.path.join(folder_dir, str(textFile)), 'r')
        lines = file1.readlines()
        
        print(textFile, len(lines), idx)
        stringArray = []
        for line in lines:
            
            stringArray.append(line.strip())
        try:
            
            dset = hf.create_dataset(str(idx), data=stringArray)
            
        except Exception as e:

            print(e)
            print('Spams', textFile) 
            spamFiles.append(textFile)
        idx = idx + 1

print("final data file")
save_path = './textFInal.hdf5'
hf = h5py.File(save_path, 'w')
idx = 0
for textFile in os.listdir(folder_dir):
    if (textFile.endswith(".txt") and textFile not in spamFiles):
        file1 = open(os.path.join(folder_dir, str(textFile)), 'r')
        lines = file1.readlines()
        
        print(textFile, len(lines), idx)
        stringArray = []
        for line in lines:
            
            stringArray.append(line.strip())
        try:
            
            dset = hf.create_dataset(str(idx), data=stringArray)
            idx = idx + 1
        except Exception as e:

            print(e)
            print('Spams', textFile) 
            spamFiles.append(textFile)
        
        

  


# ds = hf['1']


# data = np.array(ds[0:2])

# for string in data:
#     print(string.decode("utf-8") )
