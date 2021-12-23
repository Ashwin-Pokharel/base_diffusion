#This model will contain the utility functions for a simple diffusion model
import math
import numpy as np
import torch as th
import enum 

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


#get the KL divergence between normal distributions parametrized by means and log variance
def kl_normal(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * ( -1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) + ((mean2 - mean1)**2) / th.exp(logvar2))


#get a simple linear beta schedule
def get_linear_beta_schedule(beta_start, beta_end, num_diffusion_sets):
    return np.linspace(beta_start, beta_end, num_diffusion_sets , dtype=np.float64)


#class for training and sampling from a diffusion model
class DiffusionModel:
    """
    
    the variance for this model will be fixed to beta
    :param beta: beta scheduler being used
    
    """


    def __init__(self , * , betas , model_var_type , model_mean_type):
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be a 1D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])
        self.model_var_type = model_var_type
        self.model_mean_type = model_mean_type

        alphas = 1. - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)

         # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1. - self.alphas_cumprod)



    #function to get the distribution q(x_t | x_0).
    def q_mean_variance(self , x_start , t):
        """
        Get the distribution q(x_t | x_0).
        
        :param x_start: the tensor for noiseless input (x[0] or inital image)
        :param t: the number of diffusion step-1. meaning 0 = 1 steps
        :return A tuple(mean , variance , log_variance)
        
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod , t , x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod , t , x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod , t , x_start.shape)

        return mean , variance , log_variance

    #take in the  the base sample and then add noise to it
    def q_sample(self , x_start , t , noise = None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """

        if noise is None:
            noise = th.rand_like(x_start)
        assert noise.shape == x_start.shape
        return _extract_into_tensor(self.sqrt_alphas_cumprod , t , x_start.shape) +  _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod , t , x_start.shape) * noise



    #calculate the mean and variance (which is set to beta) for the posterior given current diffusion
    def q_posterior_mean_variance(self , x_start , x_t , t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)

        :param x_start: the inital data batch (x_0)
        :param x_t: the diffusion at given t
        :param t: current timestep
        :return: A tuple(posterior_mean , posterior_variance , posterior_log_variance_clipped)

        """

        assert x_start.shape == x_t.shape
        posterior_mean = _extract_into_tensor(self.posterior_mean_coef1 , t , x_t.shape) * x_start + _extract_into_tensor(self.posterior_mean_coef2 , t , x_t.shape) * x_t
        posterior_variance = _extract_into_tensor(self.posterior_variance , t , x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped , t , x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped[0] == x_start.shape[0])

        return posterior_mean , posterior_variance , posterior_log_variance_clipped
    

#get the models output for mean and variance ( this is what the nueral network is learning so it will be used in the loss function) (aka predicted outputs)
    def p_mean_variance(self , model , * , x , t , clip_denoised:bool):
        """
        

        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :return: a tuple with (mean , variance , log_variance , pred_xstart) (last one is maybe)
        """
        B, H, W, C = x.shape
        assert t.shape == [0]
        model_output = model(x , t)
        model_variance , model_log_variance = {
            ModelVarType.FIXED_SMALL : (self.posterior_variance , self.posterior_log_variance_clipped),
            ModelVarType.FIXED_LARGE : (np.append(self.posterior_variance[1], self.betas[1:]) , np.log(np.append(self.posterior_variance[1] , self.betas[1:]))
            ),
        }[self.model_var_type]

        model_variance = _extract_into_tensor(model_variance , t , x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance , t , x.shape)


        
        _maybe_clip = lambda x_: (th.clamp(x_, -1., 1.) if clip_denoised else x_)
        if self.model_mean_type == ModelMeanType.PREVIOUS_X: #model is prediction (x_t-1 | x_t)
            pred_xstart = _maybe_clip(self._predict_xstart_from_xprev(x_t=x , t=t , xprev= model_output))
            model_mean = model_output
        elif self.model_mean_type == ModelMeanType.START_X: #model is predicting x_0 (original sample)
            pred_xstart = _maybe_clip(model_output)
            model_mean , _ , _ = self.q_posterior_mean_variance(x_start=pred_xstart , x_t = x , t=t)
        elif self.model_mean_type == ModelMeanType.EPSILON: #predicitng the epsilon (modeling the noise)
            pred_xstart = _maybe_clip(self._predict_xstart_from_eps(x_t =x , t = t , eps=model_output))
            model_mean , _ , _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x , t=t)

        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)

        return model_mean , model_variance , model_log_variance , pred_xstart
    

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        Given the x_t (noise at given step t) get the epsilon at step  t f

        Args:
            x_t ([type]): [description]
            t ([type]): [description]
            pred_xstart ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)



    def _predict_xstart_from_eps(self, x_t, t, eps):
            assert x_t.shape == eps.shape
            return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
            )



    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )
#============================Sampling From the model==========================================================================================================
    



#extract values from 1-d numpy array for a batch of indexs: 
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)