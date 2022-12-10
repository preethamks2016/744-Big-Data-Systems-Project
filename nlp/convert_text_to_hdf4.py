import h5py
import numpy as np
import os
import io


save_path = './text.hdf5'
hf = h5py.File(save_path, 'w') # open a hdf5 file
folder_dir = "./text_data"

idx = 0
for textFile in os.listdir(folder_dir):
 
    if (textFile.endswith(".txt")):
        file1 = open(os.path.join(folder_dir, str(textFile)), 'r')
        lines = file1.readlines()
        print(len(lines))
        stringArray = []
        for line in lines:
            #print(line.strip())
            stringArray.append(line.strip())

        dset = hf.create_dataset(str(idx), data=stringArray)
        idx=idx+1

  


# ds = hf['1']


# data = np.array(ds[0:2])

# for string in data:
#     print(string.decode("utf-8") )

