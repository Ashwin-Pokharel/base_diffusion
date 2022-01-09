#This will contain the training code for the diffusion model
import functools
import os
import utils
import model
import nn
import numpy as np
import torch as th
from torch.optim import AdamW
from model import TimeUnet
from functools import partial
from torchinfo import summary
from dataloader import _list_image_files_recursively , ImageDataset, load_data
import cv2 as cv2

INITIAL_LOG_LOSS_SCALE = 20.0


class UniformSampler():
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights
    
    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.
        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights

class TrainLoop:
    def __init__(self, model: model.TimeUnet , diffusion , data, batch_size , micro_batch,  lr , weight_decay=0.0, ) -> None:
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = micro_batch if micro_batch > 0 else batch_size
        self.step = 0
        self.sampler = UniformSampler(diffusion)
        self.lr = lr 
        self.weight_decay = weight_decay
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    
    def run_step(self , batch):
        self.forward_backward(batch)
            
    
    def forward_backward(self, batch):
        self.opt.zero_grad()
        t , weights = self.sampler.sample(batch.shape[0], batch.device)
        compute_losses = partial(self.diffusion.training_losses , self.model , batch , t)
        losses = compute_losses()
        losses = (losses * weights).mean()
        print("type of loss {0}, data {1}".format(type(losses), losses))
        losses.backward()
        self.opt.step()
    
    def save(self):
        model_dict = self.model_dict()
        th.save(model_dict , "diffusion_unet")
       
       
        
if __name__ == '__main__':
    
    model =  TimeUnet(image_size=(64 , 64) , in_channels=1 , model_channels=128, out_channels=1 , num_res_blocks=1, channel_mult=(1 , 2 , 3 , 4) , attention_resolutions=(64,32,16,8), num_heads_upsample=1)
    model_var = utils.ModelVarType.FIXED_SMALL
    model_mean = utils.ModelMeanType.PREVIOUS_X
    loss_type = utils.LossType.KL
    beta = utils.get_linear_beta_schedule(1000)
    diffusion = utils.DiffusionModel(betas=  beta ,model_var_type= model_var ,model_mean_type= model_mean, loss_type= loss_type)
    #print(diffusion)
    print(model)
    '''
    data_path = "/Users/apokhar/Desktop/personal/diffusion_base/images/train/sad/"
    images = _list_image_files_recursively(data_path)
    dataloader = load_data(data_path , 1 , False)
    trainingLoop = TrainLoop(model , diffusion , dataloader , 1 , 1, .001)
    data = next(iter(dataloader))
    #print(data.shape)
    trainingLoop.run_step(data)
    '''
    
    
    
    
    
        
        
        
        
        


