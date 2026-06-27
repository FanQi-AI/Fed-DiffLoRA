import os
from typing import Optional

import torch


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return _DTYPE_MAP.get(str(dtype), torch.float32)


def get_model(
    model_path: str,
    weight_dtype="float32",
    device: str = "cpu",
):
    """加载 SD1.5 pipeline。"""
    from diffusers import StableDiffusionPipeline

    if not os.path.exists(model_path):
        raise ValueError(f"提供的模型路径不存在: {model_path}")

    print(f"加载 Stable Diffusion 1.5 模型: {model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=resolve_dtype(weight_dtype)
    ).to(device)
    return pipe
