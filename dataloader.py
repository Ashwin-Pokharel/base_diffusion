import os
from PIL import Image, ImageOps
import cv2 
import numpy as np
import torch
from torch.utils.data import Dataset , DataLoader
import blobfile as bf
import cv2
import random
from torchvision import transforms
  
# define custom transform
# here we are using our calculated
# mean & std


#this file will be responsible for function for data loading , need to apply random transformation among other things. 
def cycle(dl):
    while True:
        for data in dl:
            yield data


def load_data(dataset_directory,batch_size, shuffle):
    if not dataset_directory:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(dataset_directory)
    dataset = ImageDataset(all_files)
    return cycle(DataLoader(dataset , batch_size , shuffle, drop_last=True))




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
    def __init__(self , image_paths, random_flip = True, image_size=(48 , 48)):
        self.image_paths = image_paths
        self.random_flip = random_flip
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self , index):
        path = self.image_paths[index]
        img = Image.open(path).convert("L")
        #img = cv2.resize(img , (64 , 64))
        return self.transform(img)
    
    def getPath(self , index):
        path = self.image_paths[index]
        return path


if __name__ == '__main__':
    print("at data.py ")
    data_path = "/Users/apokhar/Desktop/personal/diffusion_base/images/sad_training/"
    images = _list_image_files_recursively(data_path)
    dataset = ImageDataset(images)
    img = dataset.__getitem__(0)

      

