"""风格 LoRA 的 alpha 训练 
"""
import os
from typing import Dict, List

import torch
from torch import nn, optim
from PIL import Image
from torchvision import transforms

from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from transformers import CLIPTokenizer

from . import _paths  
from Lora import LoRAModule
from .models import resolve_dtype


def _canonical_alpha_name(param_name: str, lora_filenames: List[str]) -> str:
    name = param_name.replace(".alpha_model.weight", "").replace(".alpha", "")
    for fn in lora_filenames:
        safe = fn.replace(".", "_")
        for tag in ("_te_", "_unet_"):
            prefix = safe + tag
            if name.startswith(prefix):
                return tag.strip("_") + "_" + name[len(prefix):]
    for tag in ("_unet_", "_te_"):
        idx = name.find(tag)
        if idx != -1:
            return tag.strip("_") + "_" + name[idx + len(tag):]
    return name


def _encode_images_to_latents(pipe, image_paths, device, weight_dtype):
    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    imgs = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    batch = torch.stack(imgs).to(device=device, dtype=weight_dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(batch).latent_dist.sample() * 0.18215
    return latents


def train_style_alphas(conf, spec) -> Dict[str, float]:
    
    style_files = spec.style_lora_files
    if not style_files:
        raise ValueError(
            f"客户端 {spec.client_id} 没有风格 LoRA 文件: {spec.style_lora_dir}"
        )

    device = conf.device
    weight_dtype = resolve_dtype(conf.weight_dtype)

    # 加载风格 LoRA state_dicts (放 CPU, 避免显存峰值)
    lora_filenames = [os.path.basename(f) for f in style_files]
    lora_state_dicts = [(os.path.basename(f), load_file(f, device="cpu")) for f in style_files]

    pipe = StableDiffusionPipeline.from_pretrained(
        conf.model_path, torch_dtype=weight_dtype
    ).to(device)
    loramodel = LoRAModule(pipe.text_encoder, pipe.unet, lora_state_dicts)

    param_groups = loramodel.prepare_optimizer_params()
    if not param_groups or not param_groups[0]["params"]:
        raise ValueError(f"客户端 {spec.client_id}: 未找到可优化的 alpha 参数。")
    optimizer = optim.Adam(param_groups, lr=conf.learning_rate)
    loss_fn = nn.MSELoss()

    tokenizer = CLIPTokenizer.from_pretrained(conf.model_path, subfolder="tokenizer")

    pipe.unet.train()
    pipe.text_encoder.train()
    pipe.scheduler.set_timesteps(50)
    timesteps = pipe.scheduler.timesteps.to(device)

    # 准备本地训练数据: 用客户端内容图像 + prompt
    image_paths = spec.image_paths
    prompt = spec.content_prompt or ""

    for epoch in range(conf.local_epochs):
        optimizer.zero_grad()

        if image_paths:
            # 用本地内容图像编码的 latent 作为训练目标
            bs = min(conf.batch_size, len(image_paths))
            start = (epoch * bs) % len(image_paths)
            batch_paths = image_paths[start:start + bs]
            if not batch_paths:
                batch_paths = image_paths[:bs]
            init_latents = _encode_images_to_latents(
                pipe, batch_paths, device, weight_dtype
            )
        else:
            # 客户端没有内容图像时, 用随机 latent 兜底
            init_latents = torch.randn(
                (conf.batch_size, pipe.unet.config.in_channels,
                 pipe.unet.config.sample_size, pipe.unet.config.sample_size),
                device=device, dtype=weight_dtype,
            )

        bs = init_latents.shape[0]
        t = timesteps[torch.randint(0, len(timesteps), (bs,), device=device)]
        noise = torch.randn_like(init_latents)
        noisy = pipe.scheduler.add_noise(init_latents, noise, t)

        text_in = tokenizer(
            [prompt] * bs, padding="max_length",
            max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",
        )
        emb = pipe.text_encoder(text_in.input_ids.to(device))[0].to(dtype=weight_dtype)
        noise_pred = pipe.unet(noisy, t, encoder_hidden_states=emb).sample
        loss = loss_fn(noise_pred, noise)
        loss.backward()
        optimizer.step()
        print(f"  [client {spec.client_id}] alpha epoch {epoch} loss={loss.item():.6f}")

    # 导出 alpha
    alpha: Dict[str, float] = {}
    for name, param in loramodel.named_parameters():
        if "alpha" in name and param.requires_grad:
            cname = _canonical_alpha_name(name, lora_filenames)
            alpha[cname] = float(param.data.to(torch.float32).flatten()[0].item())

    del pipe, loramodel
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return alpha