#This will contain the training code for the diffusion model
import functools
import os
from .nn import update_ema

INITIAL_LOG_LOSS_SCALE = 20.0
