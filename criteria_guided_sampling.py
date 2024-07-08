# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import inspect
from typing import List, Optional, Union
import numpy as np
import random
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from torchmetrics.image import StructuralSimilarityIndexMeasure

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)

import clip

seed = 2024
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


class VSGuidedDiffusion(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        clip_model: CLIPModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler],
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            clip_model=clip_model,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        set_requires_grad(self.text_encoder, False)
        set_requires_grad(self.clip_model, False)
        self.normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]]="auto"):
        if slice_size == "auto":
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        self.unet.set_attention_slice(None)

    def add_noise(self, img, noise, up_to_t):
        encoded = self.encode(img)
        latents = self.scheduler.add_noise(
            encoded, noise, timesteps=up_to_t).to(self.device)
        return latents

    def encode(self, img):
            with torch.no_grad():
                latent = self.vae.encode(
                    transforms.ToTensor()(img).unsqueeze(0).to(self.device).type(self.vae.dtype) * 2 - 1)
                latent = self.vae.config.scaling_factor * latent.latent_dist.sample()
            return latent

    def process_image(self, sample):
        # apply clip pre-processing
        sample = 1 / self.vae.config.scaling_factor * sample
        image = self.vae.decode(sample).sample
        image = (image / 2 + 0.5)
        image = image - image.min().detach()
        image = image / image.max().detach()

        torch_preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
            transforms.CenterCrop(size=(224, 224)),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),])
        image = torch_preprocess(image)
        return image

    def get_F_M(self, M, F, f):
        F_ = torch.cat((F, f))
        m = torch.mm(F_, f.T)
        M_ = torch.cat((M, m[:-1].T))
        M_ = torch.cat((M_, m), dim=1)
        return F_, M_

    def get_rank(self, M):
        U, _, VT = torch.linalg.svd(M)
        S = torch.diag(torch.mm(U.T, torch.mm(M, VT.T)))
        S = S / S.sum()
        entropy = -torch.sum(S * torch.log(S))
        rank = torch.exp(entropy)
        return rank

    @torch.enable_grad()
    def cond_fn(
        self,
        latents,
        timestep,
        index,
        txt_embd,
        noise_pred_original,
        criteria,
        criteria_guidance_scale,
        num_cutouts,
        prompt,
        clip_for_guidance,
        region,
        F_M,
        F_M_real,
        beta,
        regions_list
    ):
        latents = latents.detach().requires_grad_()
        noise_pred = self.unet(latents, timestep, encoder_hidden_states=txt_embd, return_dict=False)[0]

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        fac = torch.sqrt(beta_prod_t)
        sample = pred_original_sample * (fac) + latents * (1 - fac)

        image = self.process_image(sample)
        features_ = clip_for_guidance.encode_image(image).to(torch.float32)
        features_ = features_ / features_.norm(2, dim=1, keepdim=True)

        if F_M is not None:
            _, M_ = self.get_F_M(M=F_M[1], F=F_M[0], f=features_)
            rank_fake = self.get_rank(M_ / M_.shape[0])

            _, M_real_ = self.get_F_M(M=F_M_real[1], F=F_M_real[0], f=features_)
            rank_real = self.get_rank(M_real_ / M_real_.shape[0])

            rank = rank_fake - beta * rank_real
            grads = torch.autograd.grad(rank, latents)[0]
            grads_same_scale = grads / (grads.norm(2).detach() + 1e-8) * latents.norm(2).detach()
            grads = grads_same_scale * criteria_guidance_scale

        else:
            grads = 0.0
        if timestep.item() == 1:
            if F_M is None:
                F_M = [features_.detach(), torch.mm(features_, features_.T)]
            else:
                print(rank)
                # update F if reached clean sample
                F_M = self.get_F_M(M=F_M[1], F=F_M[0], f=features_.detach())

        return grads, F_M

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        eta=0.0,
        criteria='clip_loss',
        criteria_guidance_scale=0,
        num_cutouts=4,
        generator=None,
        latents=None,
        output_type="pil",
        return_dict=True,
        clip_for_guidance=None,
        guidance_freq=1,
        region='',
        F_M=None,
        F_M_real=None,
        beta=2,
        regions_list=None,
    ):  
        if not isinstance(prompt, str):
            raise NotImplementedError
        batch_size = 1
        if height % 8 != 0 or width % 8 != 0:
            raise NotImplementedError
        apply_cfg = guidance_scale > 1.0

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        txt_embd = self.text_encoder(text_input.input_ids.to(self.device))[0]
        txt_embd = txt_embd.repeat_interleave(num_images_per_prompt, dim=0)

        if apply_cfg:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
            unconditional_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            unconditional_embeddings = unconditional_embeddings.repeat_interleave(num_images_per_prompt, dim=0)
            txt_embd = torch.cat([unconditional_embeddings, txt_embd])

        latents_shape = (batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = txt_embd.dtype

        extra_set_kwargs = {}
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        timestep_index = 0

        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma

        extra_step_kwargs = {}
        for i, t in enumerate(self.progress_bar(timesteps_tensor[timestep_index:])):
            latent_model_input = torch.cat([latents] * 2) if apply_cfg else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=txt_embd).sample

            if apply_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if criteria_guidance_scale != 0 and clip_for_guidance:
                txt_embd_for_guidance = (
                    txt_embd.chunk(2)[1] if apply_cfg else txt_embd
                )
                if i % guidance_freq == 0:
                    grads, F_M = self.cond_fn(
                        latents,
                        t,
                        i,
                        txt_embd_for_guidance,
                        noise_pred,
                        criteria,
                        criteria_guidance_scale,
                        num_cutouts,
                        prompt,
                        clip_for_guidance,
                        region,
                        F_M,
                        F_M_real,
                        beta,
                        regions_list,
                    )
                    latents += grads
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        if output_type == 'torch':
            return image, F_M

        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            return image, F_M

        if output_type == "numpy":
            return image, F_M

        if not return_dict:
            return (image, None, F_M)
