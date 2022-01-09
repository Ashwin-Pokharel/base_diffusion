import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

INITIAL_LOG_LOSS_SCALE = 20.0
 
def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)

class Trainer:
    def __init__(self , *, model , initial_lg_loss_scale):
        self.model = model 
        self.initial_lg_loss_scale = initial_lg_loss_scale
        self.model_params = list(self.model.parameters())
    
    
    def zero_grad(self):
        zero_grad(self.model_params)
    
    def backward(self , loss: th.Tensor):
        loss.backward()
    
    def optimize(self , opt: th.optin.Optimizer):
        opt.step()
        return True
        
