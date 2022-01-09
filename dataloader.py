import os
from PIL import Image, ImageOps
import cv2 
import numpy as np
import torch
from torch.utils.data import Dataset , DataLoader
import blobfile as bf



#this file will be responsible for function for data loading , need to apply random transformation among other things. 

def load_data(dataset_directory,batch_size, shuffle):
    if not dataset_directory:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(dataset_directory)
    dataset = ImageDataset(all_files)
    return DataLoader(dataset , batch_size , shuffle)




def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self , image_paths):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self , index):
        path = self.image_paths[index]
        img = []
        img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img , (64 , 64))
        img = img[None , :, :, ]
        return img #returning the numpy array version of the file


if __name__ == '__main__':
    print("at data.py ")
    data_path = "/Users/apokhar/Desktop/personal/diffusion_base/images/train/sad/"
    images = _list_image_files_recursively(data_path)
    dataset = ImageDataset(images)
    print(images[0])
    print(dataset.__getitem__(0))
      

