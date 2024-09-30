# Denoising diffusion probabilistic models (DDPM)

import torch
import math
import torch.nn.functional as F
import os
import cv2
import time
import numpy as np
from utils import tensor2img

device = "cuda:1" if torch.cuda.is_available() else "cpu"


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps):
    return betas_for_alpha_bar(
        timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )


# Create a beta schedule that discretizes the given alpha_t_bar function
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


class GaussianDiffusion:
    def __init__(
            self,
            timesteps=2000,
            beta_schedule='cosine'
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )

    # Get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_1, x_2, t, noise=None, return_noise=False):
        if noise is None:
            noise = torch.randn_like(x_1)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_1.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_1.shape)

        if return_noise:
            return sqrt_alphas_cumprod_t * x_1 + sqrt_one_minus_alphas_cumprod_t * noise, sqrt_alphas_cumprod_t * x_2 + sqrt_one_minus_alphas_cumprod_t * noise, noise
        else:
            return sqrt_alphas_cumprod_t * x_1 + sqrt_one_minus_alphas_cumprod_t * noise, sqrt_alphas_cumprod_t * x_2 + sqrt_one_minus_alphas_cumprod_t * noise

    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # Compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # Compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, sourceImg1, sourceImg2, x_t1, x_t2, x_t, t, concat_type, clip_denoised=True):

        input = torch.cat([sourceImg1, sourceImg2, x_t1, x_t2], dim=1)
        pred_noise, fusion_mask = model(input, t)
        imgf = fusion_mask * x_t1 + ((torch.ones(x_t1.shape).to(device) - fusion_mask) * x_t2)

        # x_t = torch.max(imgf, x_t)
        x_t = (imgf + x_t)/2

        # Get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        x_recon1 = self.predict_start_from_noise(x_t1, t, pred_noise)
        x_recon2 = self.predict_start_from_noise(x_t2, t, pred_noise)

        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
            x_recon1 = torch.clamp(x_recon1, min=-1., max=1.)
            x_recon2 = torch.clamp(x_recon2, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t) # Here the input is imgf
        model_mean1, posterior_variance, posterior_log_variance1 = self.q_posterior_mean_variance(x_recon1, x_t1, t)
        model_mean2, posterior_variance, posterior_log_variance2 = self.q_posterior_mean_variance(x_recon2, x_t2, t)
        return model_mean, posterior_log_variance, model_mean1, posterior_log_variance1, model_mean2, posterior_log_variance2

    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, model, sourceImg1, sourceImg2, x_t1, x_t2, x_t, t, concat_type, add_noise, clip_denoised=True):
        # predict mean and variance
        model_mean, model_log_variance, model_mean1, model_log_variance1, model_mean2, model_log_variance2 = self.p_mean_variance(
            model, sourceImg1, sourceImg2, x_t1, x_t2, x_t, t, concat_type, clip_denoised=clip_denoised)

        # Random noise is added except for t=0 steps
        if add_noise:
            noise = torch.randn_like(x_t1)
            # no noise when t == 0
            nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t1.shape) - 1))))
            # Compute x_{t-1}
            pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            pred_img1 = model_mean1 + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            pred_img2 = model_mean2 + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            return pred_img, pred_img1, pred_img2
        else:
            return model_mean, model_mean1, model_mean2

    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, sourceImg1, sourceImg2, concat_type, add_noise, log_info):
        step, valid_step_sum, num, generat_imgs_num = log_info
        log_step = 100

        # Start from pure noise (for each example in the batch)
        img1 = torch.randn(sourceImg1.shape, device=device)  # 推理阶段的输入是纯噪声
        img2 = torch.randn(sourceImg1.shape, device=device)
        imgs = torch.randn(sourceImg1.shape, device=device)

        # reverse process
        for i in reversed(range(0, self.timesteps)):
            if i % log_step == 0:
                now_time = time.strftime('%Y%m%d_%H%M%S')
                print(f"[valid step] {int((step - 1) / sourceImg1.shape[0]) + 1}/{valid_step_sum}    "
                      f"[generate step] {num + 1}/{generat_imgs_num}    "
                      f"[reverse process] {i}/{self.timesteps}    "
                      f"[time] {now_time}")
            t = torch.full((sourceImg1.shape[0],), i, device=device, dtype=torch.long)
            imgs, img1, img2 = self.p_sample(model, sourceImg1, sourceImg2, img1, img2, imgs, t, concat_type, add_noise)
        return imgs

    # Sample new images
    @torch.no_grad()
    def sample(self, model, sourceImg1, sourceImg2, add_noise, concat_type, model_name, model_path,
               generat_imgs_num, step, timestr, valid_step_sum, dataset_name, img_cr, img_cb):
        extension_list = ["jpg", "tif", "png", "jpeg"]
        for num in range(generat_imgs_num):
            log_info = [step, valid_step_sum, num, generat_imgs_num]
            imgs = self.p_sample_loop(model, sourceImg1, sourceImg2, concat_type, add_noise, log_info)
            print(imgs.shape)
            for i in range(imgs.shape[0]):
                img_id = step + i
                dirPath = os.path.join("generate_imgs",
                                       dataset_name,
                                       timestr,
                                       model_name,
                                       )

                # Save images in multiple formats
                image = tensor2img(imgs[i])

                # ********* convert Ycbcr to RGB **************
                img_cr = img_cr.cpu().numpy().squeeze()
                img_cb = img_cb.cpu().numpy().squeeze()
                img = image[:, :, np.newaxis]
                img_cr = img_cr[:, :, np.newaxis].astype(np.uint8)
                img_cb = img_cb[:, :, np.newaxis].astype(np.uint8)
                imgs = np.concatenate((img, img_cr, img_cb), axis=2)
                image = cv2.cvtColor(imgs, cv2.COLOR_YCR_CB2RGB)
                image = image.astype(np.uint8)

                for extension in extension_list:
                    subdirPath = os.path.join(dirPath, extension + "_imgs")
                    if not os.path.exists(subdirPath):
                        os.makedirs(subdirPath)

                    # valid log
                    valid_log_path = os.path.join(subdirPath, "valid_log.txt")
                    valid_log = open(valid_log_path, "w")
                    valid_log.write(f"time: {timestr} \n")
                    valid_log.write(f"model_path: {model_path} \n")

                    # Save imgs
                    if generat_imgs_num == 1:
                        if img_id < 10:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_0" + str(
                                                             img_id) + "." + extension)
                        else:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_" + str(
                                                             img_id) + "." + extension)
                        cv2.imwrite(img_file_path, image)
                    else:
                        if img_id < 10:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_0" + str(
                                                             img_id) + "_num" + str(
                                                             num) + "." + extension)
                        else:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_" + str(
                                                             img_id) + "_num" + str(
                                                             num) + "." + extension)
                        cv2.imwrite(img_file_path, image)

    # Compute train losses
    def train_losses(self, model, sourceImg1, sourceImg2, t, loss_scale):
        noise = torch.randn_like(sourceImg1)
        # Get x_t
        x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)

        # Concatenation is performed in the channel dimension
        input = torch.cat([sourceImg1, sourceImg2, x1_noisy, x2_noisy], dim=1)

        predicted_noise, mask = model(input, t)
        imgf = mask * x1_noisy + ((torch.ones(mask.shape).to(device) - mask) * x2_noisy)
        # print('shiyu***********************************')
        # print(predicted_noise.shape)
        # print(mask.shape)

        # loss
        # noise_loss = loss_scale * F.mse_loss(noise, predicted_noise)
        noise_loss = loss_scale * torch.norm(noise-predicted_noise, p=1) /(256*256)
        fusion_loss = (torch.norm(imgf-x1_noisy, p=1) + torch.norm(imgf-x2_noisy, p=1)) / (256*256)
        # fusion_loss = F.mse_loss(imgf, x1_noisy) + F.mse_loss(imgf, x2_noisy)
        assert predicted_noise.shape == noise.shape
        return noise_loss, fusion_loss
