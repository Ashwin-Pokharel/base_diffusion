#this class will contain code to actually generate and save faces from the model
import functools
import os

from numpy.lib.function_base import average
import utils
import model
import nn
import numpy as np
import torch as th
from torch.optim import AdamW
from model import TimeUnet
from functools import partial
from torchinfo import summary
import cv2 as cv2
from tqdm import tqdm , trange
import wandb
import warnings
import logging
from PIL import Image
import time



def generate_sample(model_path: str, num_samples:int, batch: int, image_size, output_path:str , save_individually:bool):
    try:
        model = TimeUnet(image_size=image_size , in_channels=1 , model_channels=128, out_channels=1 , num_res_blocks=2, channel_mult=(1 , 2 , 2 , 2) , attention_resolutions=(16,), num_heads_upsample=1, dropout=0.1)
        opt = AdamW(model.parameters(), lr=.0001, weight_decay=0)
        model_var = utils.ModelVarType.FIXED_SMALL
        model_mean = utils.ModelMeanType.PREVIOUS_X
        loss_type = utils.LossType.KL
        beta = utils.get_linear_beta_schedule(1000)
        diffusion = utils.DiffusionModel(betas=  beta ,model_var_type= model_var ,model_mean_type= model_mean, loss_type= loss_type)
        loaded_model = th.load(model_path)
        model.load_state_dict(loaded_model['model_state_dict'])
        opt.load_state_dict(loaded_model['optimizer_state_dict'])
        model.eval()
        logging.basicConfig(filename="sampling_log.txt", encoding='utf-8', level=logging.DEBUG)
        samples = []
        while len(samples) < num_samples:
            sample = diffusion.p_sample_loop(model , (batch , 1 , 48, 48) , progress=True)
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            samples.append(sample)
        
        if save_individually:
            for index , image in enumerate(samples):
                path = f"sampled_image_0{index}.jpeg"
                path = os.path.join(output_path , path)
                im = np.reshape(image , (image_size[0], image_size[1]))
                im = im.numpy()
                cv2.imwrite(path , im)
        else:
            images = np.concatenate(samples , axis=0)
            path = f"sampled_collection.jpeg"
            path = os.path.join(output_path, path)
            cv2.imwrite(path , images)
    except Exception as e:
        print("error")
        raise e
        



if __name__ == '__main__':
    generate_sample("/Volumes/Samsung_T5/personal/diffusion_checkpoints/diffusion_unet_epoch_6.pth", 1 , 1 , [48 , 48] , "sampled_images/" , True)       
        
        
    
    