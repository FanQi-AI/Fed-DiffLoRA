"""内容/风格融合 
"""
import os
from typing import Dict, Optional

import torch

from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

from . import _paths  # noqa: F401
from ziplora_pytorch.ziplora import ZipLoRALinearLayer
from .models import resolve_dtype


def _mean_alpha(aggregated_alpha: Dict[str, float]) -> float:
    if not aggregated_alpha:
        return 1.0
    return float(sum(aggregated_alpha.values()) / len(aggregated_alpha))


def fuse_content_style(
    conf,
    spec,
    aggregated_alpha: Dict[str, float],
    save_path: Optional[str] = None,
) -> Dict:
    """融合内容 LoRA 与风格 LoRA """
    device = conf.device
    weight_dtype = resolve_dtype(conf.weight_dtype)
    style_scale = _mean_alpha(aggregated_alpha)

    pipe = StableDiffusionPipeline.from_pretrained(
        conf.model_path, torch_dtype=weight_dtype
    ).to(device)
    unet = pipe.unet

    # 加载内容 LoRA 权重 
    content_sd = (
        load_file(spec.content_lora_path, device="cpu")
        if spec.content_lora_path and os.path.isfile(spec.content_lora_path)
        else {}
    )
    # 加载风格 LoRA 权重 
    style_files = spec.style_lora_files
    style_sd = (
        load_file(style_files[0], device="cpu")
        if style_files and os.path.isfile(style_files[0])
        else {}
    )

    fused_layers = 0
    for attn_name in unet.attn_processors.keys():
        attn_module = unet
        for n in attn_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        for part in ("to_q", "to_k", "to_v"):
            proj = getattr(attn_module, part, None)
            if proj is None or not hasattr(proj, "set_lora_layer"):
                continue
            zip_layer = ZipLoRALinearLayer(
                in_features=proj.in_features,
                out_features=proj.out_features,
                init_merger_value=1.0,            
                init_merger_value_2=style_scale,  
                device=device,
                dtype=weight_dtype,
            )
            proj.set_lora_layer(zip_layer)
            fused_layers += 1

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        from ziplora_pytorch.utils import unet_ziplora_state_dict
        torch.save(unet_ziplora_state_dict(unet), save_path)

    summary = {
        "client_id": spec.client_id,
        "content_lora": os.path.basename(spec.content_lora_path) if spec.content_lora_path else "<none>",
        "num_style_layers": len(aggregated_alpha),
        "style_scale_from_alpha": round(style_scale, 6),
        "fused_attn_layers": fused_layers,
    }

    del pipe, unet
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return summary