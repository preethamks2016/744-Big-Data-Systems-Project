import h5py
import numpy as np
import os
from PIL import Image
# import cv2

save_path = './test.hdf5'
img_path = '1.jpeg'
print('image size: %d bytes'%os.path.getsize(img_path))
hf = h5py.File(save_path, 'w') # open a hdf5 file
ds = hf.create_dataset('data', (2, 256, 256, 3), dtype='float32', compression="gzip", compression_opts=9)
img_np = np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
print(img_np.shape)

ds[0] = img_np
ds[1] = img_np

# ds[i] = ith image


# reading
img0 = hf['data'][0:2]
print(img0.shape)

print(type(img0))

hf.close()  # close the hdf5 file
print('hdf5 file size: %d bytes'%os.path.getsize(save_path))

# print(type(ds[0:1]))

# data = np.array(ds[0])
# # img = Image.open(data)
# # img.show()