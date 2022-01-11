#This will contain the training code for the diffusion model
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
from dataloader import _list_image_files_recursively , ImageDataset, load_data
import cv2 as cv2
from tqdm import tqdm , trange
import wandb
import warnings
import logging

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
    def __init__(self, model: model.TimeUnet , diffusion , data, batch_size , micro_batch,  lr , weight_decay=0.0, num_epochs=10, checkpoint=None , checkpoint_epoch=0 ) -> None:
        self.model = model
        self.diffusion = diffusion
        self.dataloader = data
        self.batch_size = batch_size
        self.microbatch = micro_batch if micro_batch > 0 else batch_size
        self.step = 0
        self.sampler = UniformSampler(diffusion)
        self.lr = lr 
        self.weight_decay = weight_decay
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.num_epochs = num_epochs
        self.current_epochs = 0
        self.running_loss = []
        self.current_loss = None
        self.run = wandb.init(project="diffusion_base", entity="ashwin_pokharel", reinit=True)
        wandb.config.update({
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        }) 
        logging.basicConfig(filename="training_log.txt", encoding='utf-8', level=logging.DEBUG)
            
    
    
    def run_loop(self):
        logging.debug("Train loop starts here")
        wandb.run.name = "diffusion_base_training_1"
        counter = 1
        try:
            with trange(self.num_epochs , position=0 , unit='epoch')as pbar:
                for epoch in pbar:
                    current_epoch = 0
                    pbar.set_description(f"Epoch: {epoch}")
                    counter = 1
                    with tqdm(self.dataloader , position=1 , unit="images" , leave=False) as inner_bar:
                        for value in inner_bar:
                            self.run_step(value)
                            inner_bar.set_description(f"# images: {counter}")
                            counter += 1
                            if(counter % 500 == 0):
                                current_avg_loss = np.average(self.running_loss)
                                logging.info(f"Epoch {epoch}: {counter} images processed : average_loss is {current_avg_loss}")
                    
                    self.save_checkpoint(epoch , self.current_loss)
                    current_avg_loss = np.average(self.running_loss)
                    logging.info(f"Epoch #{epoch} finished, average loss is {current_avg_loss}")        
                    pbar.set_postfix(f"Avg Loss: {current_avg_loss}")
                    
                
                    
        except Exception as e:
            self.save_checkpoint(epoch , self.current_loss)
            raise e
        
        self.run.finish()
        self.save()
                    
                    
     
               
    def save_checkpoint(self , epoch , loss):
        path =  f"./checkpoints/diffusion_unet_epoch_{epoch}.pt"
        th.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'loss': loss,
            'average_loss': self.running_loss
        })      
            
    
    def run_step(self , batch):
        self.forward_backward(batch)
            
    
    def forward_backward(self, batch):
        self.opt.zero_grad()
        t , weights = self.sampler.sample(batch.shape[0], batch.device)
        compute_losses = partial(self.diffusion.training_losses , self.model , batch , t)
        losses = compute_losses()
        losses = (losses * weights).mean()
        losses.backward()
        loss = losses.clone().detach()
        self.current_loss = loss
        self.running_loss.append(loss)
        wandb.log({
            "loss": losses,
            "avg_loss": np.average(self.running_loss)
        })
        self.opt.step()
    
    def save(self):
        model_dict = self.model_dict()
        th.save(model_dict , "diffusion_unet.pt")
       
       
        
if __name__ == '__main__':
    
    model =  TimeUnet(image_size=(64 , 64) , in_channels=1 , model_channels=128, out_channels=1 , num_res_blocks=1, channel_mult=(1 , 2 , 3 , 4) , attention_resolutions=(64,32,16,8), num_heads_upsample=1)
    model_var = utils.ModelVarType.FIXED_SMALL
    model_mean = utils.ModelMeanType.PREVIOUS_X
    loss_type = utils.LossType.KL
    beta = utils.get_linear_beta_schedule(1000)
    diffusion = utils.DiffusionModel(betas=  beta ,model_var_type= model_var ,model_mean_type= model_mean, loss_type= loss_type)
    #print(diffusion)
    #print(model)
    data_path = "/Users/apokhar/Desktop/personal/diffusion_base/images/sad_training/"
    images = _list_image_files_recursively(data_path)
    dataloader = load_data(data_path , 1 , False)
    trainingLoop = TrainLoop(model , diffusion , dataloader , 1 , 1, .001)
    #print(data.shape)
    trainingLoop.run_loop()
    
    
    
    
    
    
        
        
        
        
        


