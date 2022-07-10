#This will contain the training code for the diffusion model
from copy import deepcopy
import functools
import os
from tkinter import N
import uuid

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
from sampler import generate_sample
from dotenv import load_dotenv
import optuna

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
    def __init__(self, model: model.TimeUnet , diffusion , data, batch_size, image_size , micro_batch,  lr , weight_decay=0.0, num_epochs=10, checkpoint=False , checkpoint_path=None, num_steps=0 ,  resume=None ) -> None:
        self.model = model
        self.diffusion = diffusion
        self.dataloader = data
        self.batch_size = batch_size
        self.image_size = image_size
        self.microbatch = micro_batch if micro_batch > 0 else batch_size
        self.current_step = 0
        self.num_steps = num_steps
        self.current_epoch = 0
        self.sampler = UniformSampler(diffusion)
        self.lr = lr 
        self.weight_decay = weight_decay
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.num_epochs = num_epochs
        self.current_epochs = 0
        self.running_loss = []
        self.current_loss = None
        load_dotenv()
        
        if "RUN_ID" in os.environ:
            self.run_id = os.getenv("RUN_ID")
        else:
            self.run_id = wandb.util.generate_id()
            with open(".env", "w") as f:
                f.write("RUN_ID={0}".format(self.run_id))
        
        self.run = wandb.init(project="diffusion_base", entity="ashwin_pokharel",id=self.run_id , resume=resume)
        wandb.config.update({
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": batch_size,
            "num_steps": num_steps,
        })
        
        logging.basicConfig(filename="training_log.txt", encoding='utf-8', level=logging.DEBUG)
        if(checkpoint == True):
            self.load_step(checkpoint_path)
            
    
    
    def run_step_loop(self):
        logging.debug("Train loop starts here")
        wandb.run.name = "diffusion_base_loss_corrected_training_2"
        running_step = self.current_step
        try:
            with trange(self.current_step , self.num_steps , position=0 , unit='steps') as pbar:
                for step in pbar:
                    running_step = step
                    pbar.set_description(f"Step: {step}")
                    image = next(self.dataloader)
                    self.run_step(image)
                    if(step % 100 == 0):
                        logging.info(f"Step {step}: current loss: {self.current_loss}")
                    
                    if(step % 500 == 0):
                        self.save_step_checkpoint(step , self.current_loss)
                        logging.info(f"Step {step}: current loss: {self.current_loss}")
                    wandb.log({
                    "loss": self.current_loss,
                    })
                    
                    pbar.set_postfix({"loss": self.current_loss})
            
            self.run.finish()
            self.save()
        except Exception as e:
            self.save_step_checkpoint(running_step , self.current_loss)
            logging.debug(e)
            raise e
                
                
    def run_loop(self):
        logging.debug("Train loop starts here")
        wandb.run.name = "diffusion_base_loss_corrected_training_1"
        counter = 1
        try:
            with trange(self.current_epoch+1 , self.num_epochs, position=0 , unit='epoch')as pbar:
                for epoch in pbar:
                    pbar.set_description(f"Epoch: {epoch}")
                    counter = 1
                    with tqdm(self.dataloader , position=1 , unit="batch" , leave=False) as inner_bar:
                        for value in inner_bar:
                            self.run_step(value)
                            inner_bar.set_description(f"Batch: {counter}")
                            counter += 1
                            if(counter % 1000 == 0):
                                current_avg_loss = np.average(self.running_loss)
                                logging.info(f"Epoch {epoch}: {counter} batch processed : average_loss is {current_avg_loss}")
                            
                    
                    self.save_checkpoint(epoch , self.current_loss)
                    current_avg_loss = np.average(self.running_loss)
                    logging.info(f"Epoch #{epoch} finished, average loss is {current_avg_loss}")        
                    pbar.set_postfix({"Avg loss": current_avg_loss})
                    self.running_loss = []
                    
                
                    
        except Exception as e:
            print(e)
            self.save_checkpoint(self.current_epoch , self.current_loss)
            raise e
        
        self.run.finish()
        self.save()
                    
    
    def save_step_checkpoint(self , current_step , loss):
        path =  "/Volumes/Samsung_T5/personal/diffusion_corrected_checkpoints/diffusion_unet_step_{0}.pth".format(current_step)
        th.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'loss': loss,
            'step': current_step,
        } , path)
        try:
            wandb.save(path, base_path="/Volumes/Samsung_T5/personal/") #updated to avoid warning
        except:
            wandb.save(path)
        try:
            generate_sample(path, 1 , 1 , [48 , 48] , "sampled_images/" , True, epoch=current_step)
            logging.info("sample generated successfully")
        except Exception as e:
            logging.debug("*"*100)
            logging.debug("ERROR GENERATING SAMPLES")
            
            
                       
    
    def load_step(self , model_path):
        loaded_model = th.load(model_path)
        self.model.load_state_dict(loaded_model['model_state_dict'])
        self.opt.load_state_dict(loaded_model['optimizer_state_dict'])
        self.current_step = loaded_model['step']
        self.current_loss = loaded_model['loss']
    
    
               
    def save_checkpoint(self , epoch , loss):
        path =  "/Volumes/Samsung_T5/personal/diffusion_corrected_checkpoints/diffusion_unet_epoch_{0}.pth".format(epoch)
        th.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'loss': loss,
            'batch': self.batch_size,
            'average_loss': self.running_loss
        } , path)
        output_Path = "/Users/apokhar/Desktop/personal/diffusion_base/sampled_images/"
        generate_sample(path, 1 , 1 , [48 , 48] , "sampled_images/" , True, epoch=epoch)
         
    def load(self, model_path):
        loaded_model = th.load(model_path)
        self.model.load_state_dict(loaded_model['model_state_dict'])
        self.opt.load_state_dict(loaded_model['optimizer_state_dict'])
        self.batch_size = loaded_model['batch']
        self.current_epoch = loaded_model['epoch']
        self.current_loss = loaded_model['loss']
            
    
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
        self.current_loss = loss.item()
        
        
        self.opt.step()
    
    def save(self):
        try:
            best_model_state = deepcopy(self.model.state_dict())
            th.save( best_model_state, "/Volumes/Samsung_T5/personal/diffusion_corrected_checkpoints/final_unet.pth")
        except Exception as e:
            th.save(self.model , "/Volumes/Samsung_T5/personal/diffusion_corrected_checkpoints/final_unet.pth")
    
        
if __name__ == '__main__':
    checkpoint_path = None
    model =  TimeUnet(image_size=( 48 , 48) , in_channels=1 , model_channels=64, out_channels=1 , num_res_blocks=2, channel_mult=(1, 2, 4, 8) , attention_resolutions=(16,), num_heads_upsample=1, dropout=0.2)
    model_var = utils.ModelVarType.FIXED_SMALL
    model_mean = utils.ModelMeanType.START_X
    loss_type = utils.LossType.KL
    beta = utils.get_linear_beta_schedule(1000)
    diffusion = utils.DiffusionModel(betas=  beta ,model_var_type= model_var ,model_mean_type= model_mean, loss_type= loss_type)
    #print(diffusion)
    #print(model)
    data_path = "/Users/apokhar/Desktop/personal/diffusion_base/images/sad_training/"
    images = _list_image_files_recursively(data_path)
    batch = 48
    dataloader = load_data(data_path , batch , True)
    trainingLoop = TrainLoop(model , diffusion , dataloader , batch, (48 , 48) , 0 , 1e-5, num_epochs=100, checkpoint=False , checkpoint_path=checkpoint_path, num_steps=700000, resume=False)
    #print(data.shape)
    trainingLoop.run_step_loop()
    
    
    
    
    
    
        
        
        
        
        


