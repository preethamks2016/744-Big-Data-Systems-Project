import h5py
import numpy as np
import os
from PIL import Image

save_path = './testCompOpts4.hdf5'
folder_dir = "../data/publaynet/train/"

idx = 0
hf = h5py.File(save_path, 'w') # open a hdf5 file
ds = hf.create_dataset('data', (25000, 700, 600, 3), dtype='float32', compression="gzip", compression_opts=4)
count = 0
for image in os.listdir(folder_dir):
    if (image.endswith(".jpg")):
       
        with open(os.path.join(folder_dir, str(image)), 'rb') as img_f:
            img = Image.open(img_f).convert('RGB')
            print(img.size, count)
            if(img.size[0] >= 600 and img.size[1] >= 700):
                img_np = np.array(img.resize((600,700)))
                ds[count] = img_np
                count = count+1
                if(count == 1000):
                    break
hf.close()
print('Done', count)

            
            
