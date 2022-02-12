from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchinfo import summary

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

print(model)

diffusion = GaussianDiffusion(
    model,
    image_size = 48,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

trainer = Trainer(
    diffusion,
    "/Users/apokhar/Desktop/personal/diffusion_base/images/sad_training/",
    train_batch_size = 32,
    image_size=48,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False                       # turn on mixed precision training with apex
)



