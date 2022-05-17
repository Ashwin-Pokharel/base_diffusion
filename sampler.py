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
from torchvision import utils as torchVisionUtils



def generate_sample(model_path: str, num_samples:int, batch: int, image_size:tuple[int , int] ,  output_path:str , save_individually:bool, epoch=None):
    try:
        model =  TimeUnet(image_size=image_size, in_channels=1 , model_channels=64, out_channels=1 , num_res_blocks=2, channel_mult=(1, 2, 4, 8) , attention_resolutions=(16,), num_heads_upsample=1, dropout=0.2)
        opt = AdamW(model.parameters(), lr=0.00002, weight_decay=0)
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
            start_time = time.perf_counter()
            sample = diffusion.p_sample_loop(model , (batch , 1 , 48, 48) , progress=True)
            elapsed = time.perf_counter() - start_time
            print(f"diffusion sampled in {elapsed:0.2f} seconds.")
            sample = (sample + 1) * 0.5
            samples.append(sample)
        
        if save_individually:
            for index , image in enumerate(samples):
                path = f"sampled_image_0{index}.jpeg"
                if epoch != None:
                    path = f"sampled_image_0{index}_{epoch}.jpeg"
                path = os.path.join(output_path , path)
                torchVisionUtils.save_image(image, path)
       
    except Exception as e:
        print("error")
        raise e
        



if __name__ == '__main__':
    generate_sample("/Volumes/Samsung_T5/personal/diffusion_corrected_2_checkpoints/diffusion_unet_step_12000.pth", 1 , 1 , (48 , 48) , "sampled_images/" , True)       
        

    
    