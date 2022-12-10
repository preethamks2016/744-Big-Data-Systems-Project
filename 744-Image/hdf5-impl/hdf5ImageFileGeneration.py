import os
from os import listdir

import io
import h5py
from PIL import Image
import numpy as np
import cv2
import torch
hdf5_path = './images.hdf5'
hf = h5py.File(hdf5_path, 'w') # open a hdf5 file

# get the path/directory
folder_dir = "/home/rkosgi/744Project/744-Image/publaynet/train"
idx = 0
for image in os.listdir(folder_dir):
 
    # check if the image ends with png
    if (image.endswith(".jpg")):
        # print(image)
        # print(idx)
        print(str(image))
        img_to_blur = cv2.imread(os.path.join(folder_dir, str(image)))
        print(img_to_blur.shape)
        with open(os.path.join(folder_dir, str(image)), 'rb') as img_f:
            binary_data = img_f.read()      # read the image as python binary
        binary_data_np = np.asarray(binary_data)
        dataset_name = str(idx)
        dset = hf.create_dataset(dataset_name, data=binary_data_np)
        hf.close()  # write the data to hdf5 file

        # idx=idx+1
        hdf5_path = './images.hdf5'
        hf = h5py.File(hdf5_path, 'r')
        data = np.asarray(hf[str(idx)])
        img = Image.open(io.BytesIO(data))
        size = img.size
        size = size[::-1]
        new_img = np.asarray(img.convert('RGB')).reshape([*size,3])
        img_to_blur = torch.from_numpy(new_img)

        print(img.mode, "HW",img_to_blur.size())
        # 
        # data = np.array(hf['0'])   # write the data to hdf5 file
        # img = np.asarray(io.BytesIO(data))
        # print(img.shape)
        # print(img)
        # # img = Image.open(io.BytesIO(data))
        # size = img.size
        # print(img.size)
        # # size = size[::-1]
        # new_img = np.asarray(img.convert('RGB')).reshape([*size,3])
        # print('image size:', img.size)
        #img = Image.open(io.BytesIO(data))
        # img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        #size = img.size
        #size = size[::-1]
        # print(type(img))
        #new_img = np.asarray(img.convert('RGB')).reshape([*size,3])
        #img_to_blur = torch.from_numpy(new_img)
        #r1 = torch.tensor(random.randint(0, 1))
        hf.close() 

        break
    
hf.close()




# hdf5_path = './images.hdf5'
# hf = h5py.File(hdf5_path, 'r') # open a hdf5 file

# # print(list(hf.keys()))

# # key = list(hf.keys())[3]

# # print("Key: %s" % key)

# data = np.array(hf['2'])   # write the data to hdf5 file
# img = Image.open(io.BytesIO(data))
# print('image size:', img.size)
# hf.close()  # close the hdf5 file
# img.show()