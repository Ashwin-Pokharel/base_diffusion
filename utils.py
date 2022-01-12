#This model will contain the utility functions for a simple diffusion model
import math
import numpy as np
import torch as th
import nn
import enum

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()

class LossType(enum.Enum):
    KL = enum.auto()
    MSE = enum.auto()

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

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def get_linear_beta_schedule(num_diffusion_steps):
    scale = 1000/num_diffusion_steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return _get_linear_beta_schedule(beta_start , beta_end ,num_diffusion_steps)

#get a simple linear beta schedule
def _get_linear_beta_schedule(beta_start, beta_end, num_diffusion_sets):
    return np.linspace(beta_start, beta_end, num_diffusion_sets , dtype=np.float64)


#class for training and sampling from a diffusion model
class DiffusionModel:
    """
    
    the variance for this model will be fixed to beta
    :param beta: beta scheduler being used
    
    """
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t


    def __init__(self , * , betas , model_var_type , model_mean_type, loss_type):
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])
        self.model_var_type = model_var_type
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type

        alphas = 1.0 - betas
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
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] == x_start.shape[0])

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
        B,C = x.shape[:2]
        assert t.shape == (B,)
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
    
    def p_sample(self, model , x , t ,noise=None,  clip_denoised=True , denoised_fn=None):
        """ 
        Sample x_{t-1} from the model at the given time stamp

        :param model: the model to sample from
        "param x: the current x at x_{t-1}
        :param t: the value of t , starts at 0 for the first diffusion step and then onward
        :param clip_denoised: if true, clip the x_start prediction into [-1, 1].
        :param denoised_fn: the denoising function to applied to predicted x_start before sampling
        :return: the sampled x_t
        """
        #sampled form the model
        model_mean , model_variance , model_log_variance , pred_xstart = self.p_mean_variance(model , x=x , t=t , clip_denoised=clip_denoised)
        if(noise == None):
            noise = th.randn_like(x)
        nonzero_mask = (
            (t!=0).float().view(-1 , *([1]*(len(x.shape)-1)))
        )
        sample = model_mean + nonzero_mask * th.exp(model_log_variance) * noise
        return sample , pred_xstart
        
    
    def p_sample_loop(self , model , shape , noise=None , clip_denoised=True , denoised_fn=None , device=None , progress=False):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        
        final = None
        for sample , pred_xstart in self.p_sample_loop_progressive(model , shape , noise , clip_denoised , device , progress):
            final = sample
        
        return final
    
    
    
    def p_sample_loop_progressive(self , model , shape , noise=None, clip_denoised=True , denoised_fn=None , device= None, progress=False ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over tuple where each tuple is the return value of p_sample
        """
        
        if device is None:
            device = next(model.parameters()).device #might need to come back and take a look at this
        
        assert isinstance(shape , (tuple , list))
        if noise is not None:
            img = noise 
        else:
            img = th.randn(*shape , device=device)
        
        indices = list(range(self.num_timesteps))[::-1] #going from max_timestamp-1 to 0 aka noise to x{t_0}
        if progress: 
            from tqdm.auto import tqdm
            
            indices = tqdm(indices)
        
        for i in indices:
            t = th.tensor([i]* shape, device=device)
            with th.no_grad():
                sample , pred_xstart = self.p_sample(model=model , x=img , t=t , noise=noise , clip_deoined=clip_denoised , denoised_fn=denoised_fn)
                yield sample , pred_xstart
                img = sample
        

#============================Log Liklehood sampling========================================================================================================== 

    def _vb_terms_bpd(self , model , x_start , x_t , t , clip_denoised=True):
        """
        Get a term for the variational lower-bound
        
        resulting units are in bits and not nats (look at self-information)
        
        return - a tuple
            - output: a shape[N] tensor of NLL or KL (divergences) to calculate the loss function
            - pred_xstart: the x_0 predictions
        
        """
        true_mean , true_variance , true_log_variance = self.q_posterior_mean_variance( x_start, x_t , t)
        model_mean , model_variance , model_log_variance , pred_xstart = self.p_mean_variance(model=model, x=x_t , t=t , clip_denoised=clip_denoised)
        kl = kl_normal(true_mean , true_log_variance , model_mean , model_log_variance)
        kl = nn.mean_flat(kl) / np.log(2) #convert to bits  
        deconder_nll = -discretized_gaussian_log_likelihood(x_t , means=model_mean , log_scales= 0.5 * model_log_variance)
        
        assert deconder_nll.shape == x_start.shape
        
        output = th.where((t==0 ), deconder_nll , kl)

        return output , pred_xstart    

    
    
    def training_losses(self, model , x_start , t , noise=None):
        """
         Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: losses containing a tensor of shape [N].
                
        """
        if noise is None:
            noise = th.randint_like(x_start, th.max(x_start))
            
        
        assert x_start.shape == noise.shape
        x_t = self.q_sample(x_start , t , noise)
        
        if self.loss_type == LossType.KL:
            loss, _  = self._vb_terms_bpd(model , x_start , x_t , t , clip_denoised=False)
        elif self.loss_type == LossType.MSE:
            model_output = model(x_t , self._scale_timesteps(t))
            if self.model_var_type in [ModelVarType.FIXED_LARGE, ModelVarType.FIXED_SMALL]:
                B, C = x_t.shape[:2]
                assert model_output.shape == [B , C * 2 , *x_t.shape[2:]]
                model_output , model_var_values = th.split(model_output , C , dim=1)
                
                output, pred_xstart = self._vb_terms_bpd(model , model_output , x_t , t , clip_denoised=False)
                target = {
                    ModelMeanType.PREVIOUS_X : self.q_posterior_mean_variance(x_start , x_t , t)[0],
                    ModelMeanType.START_X: x_start,
                    ModelMeanType.EPSILON: noise,
                }[self.model_mean_type]              
                assert model_output.shape == target.shape == x_start.shape
                mse = nn.mean_flat((target - output)**2)
                loss = mse 
        else:
            raise NotImplementedError(self.loss_type)
    
        return loss 
    
    def _prior_bpd(self , x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """  
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps -1] * batch_size , device=x_start.device)
        qt_mean , _ , qt_log_variance = self.q_mean_variance(x_start , t)
        kl_prior = nn.normal_kl(qt_mean , qt_log_variance , 0 , 0)
        return nn.mean_flat(kl_prior) / np.log(2) #convert to bits
    
    def calc_bpd_loop(self , model , x_start , clip_denoised=True):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a tuple containing the following quantities:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]
        
        vb = []
        x_start_mse = []
        mse = []
        for t in list(range(self.num_timesteps)[::-1]):
            t_batch = th.tensor([t] * batch_size , device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start , t_batch , noise)
            with th.no_grad():
                output , pred_xstart = self._vb_terms_bpd(model , x_start , x_t , t_batch , clip_denoised=clip_denoised)
                vb.append(output)
                x_start_mse.append(nn.mean_flat((pred_xstart - x_start)**2))
                eps = self._predict_eps_from_xstart(x_t , t_batch , pred_xstart)
                mse.append(nn.mean_flat((eps - noise)**2))
            
        vb = th.stack(vb , dim=1)
        x_start_mse = th.stack(x_start_mse , dim=1)
        mse = th.stack(mse , dim=1)
        
        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        
        return total_bpd , prior_bpd , vb , x_start_mse , mse
    
    def __str__(self):
        return ("diffusion of num_timesteps %d"%self.num_timesteps)
            
    
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
