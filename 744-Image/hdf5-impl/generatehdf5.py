import os
from os import listdir

import io
import h5py
from PIL import Image
import numpy as np

hdf5_path = './imagesCompressed.hdf5'
hf = h5py.File(hdf5_path, 'w') # open a hdf5 file

# get the path/directory
folder_dir = "/home/rkosgi/744Project/744-Image/publaynet/train"
idx = 0
for image in os.listdir(folder_dir):
 
    # check if the image ends with png
    if (image.endswith(".jpg")):
        # print(image)
        # print(idx)
        with open(os.path.join(folder_dir, str(image)), 'rb') as img_f:
            binary_data = img_f.read()      # read the image as python binary

        binary_data_np = np.asarray(binary_data)
        dataset_name = str(idx)
        dset = hf.create_dataset(dataset_name, data=binary_data_np, compression="gzip", compression_opts=9)  # write the data to hdf5 file

        idx=idx+1
    
hf.close()