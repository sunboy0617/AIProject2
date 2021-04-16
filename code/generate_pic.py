import os
import numpy as np
import cv2
from tqdm.autonotebook import tqdm

np_root = '../dataset/q1_data'
phases = ['test', 'train']

for phase in phases:
    np_name = np_root + '/' + phase + '.npy'
    pic_data =np.load(np_name)
    dataset_length = pic_data.shape[0]
    save_path = os.path.join(np_root, phase)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('Process '+phase)
    with tqdm(total = dataset_length) as pbar:
        for idx in range(dataset_length):
            temp_pic = pic_data[idx,:]
            if idx==0:
                print(temp_pic.shape)
            temp_pic = temp_pic.reshape(3,32,32).swapaxes(0,1).swapaxes(1,2)
            cv2.imwrite(os.path.join(save_path, str(idx).zfill(5)+'.jpg'), temp_pic)
            pbar.update(1)