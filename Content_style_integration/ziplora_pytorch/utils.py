import os
from typing import Optional, Dict
from huggingface_hub import hf_hub_download

import torch
from safetensors import safe_open
from diffusers import UNet2DConditionModel
from .ziplora import  ZipLoRALinearLayerInference, ZipLoRALinearLayer
import torch
from safetensors.torch import load_file
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"


import os
import torch
from safetensors.torch import safe_open
from typing import Dict, Optional
# 内容Lora
def get_lora_weights1(
    lora_name_or_path: str, subfolder: Optional[str] = None, **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Args:
        lora_name_or_path (str): huggingface repo id or folder path of lora weights
        subfolder (Optional[str], optional): sub folder. Defaults to None.
    """
    if os.path.exists(lora_name_or_path):
        if subfolder is not None:
            lora_name_or_path = os.path.join(lora_name_or_path, subfolder)
        if os.path.isdir(lora_name_or_path):
            lora_name_or_path = os.path.join(lora_name_or_path, LORA_WEIGHT_NAME_SAFE)
    else:
        lora_name_or_path = hf_hub_download(
            repo_id=lora_name_or_path,
            filename=LORA_WEIGHT_NAME_SAFE,
            subfolder=subfolder,
            **kwargs,
        )
    assert lora_name_or_path.endswith(
        ".safetensors"
    ), "Currently only safetensors is supported"
    tensors = {}
    with safe_open(lora_name_or_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors
# 风格Lora
def get_lora_weights2(
    lora_folder_path: str, subfolder: Optional[str] = None, **kwargs
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Args:
        lora_folder_path (str): 文件夹路径，包含多个 .safetensors 的 Lora 权重文件
        subfolder (Optional[str], optional): 子文件夹. Defaults to None.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]:
            外层字典 key 是 safetensors 文件名，value 是 {tensor_name: tensor} 的字典
    """
    all_tensors = {}

    if subfolder is not None:
        lora_folder_path = os.path.join(lora_folder_path, subfolder)

    if not os.path.exists(lora_folder_path):
        raise FileNotFoundError(f"Folder not found: {lora_folder_path}")

    # 遍历文件夹下所有 .safetensors 文件
    for filename in os.listdir(lora_folder_path):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(lora_folder_path, filename)
            tensors = {}
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            all_tensors[filename] = tensors

    return all_tensors


import torch
from safetensors.torch import load_file
from typing import Optional, Dict # Ensure these are imported
import torch
from safetensors.torch import load_file
from typing import Optional, Dict  # Ensure these are imported

def recursive_get_module(module, path):
    """
    递归访问模块，支持像 'to_out.0' 这种路径
    """
    parts = path.split(".")
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def unet_create_modules(root_module, state_dict):
    loras = []
    cleaned_dict = {}
    for key, value in state_dict.items():
        new_key = key

        # 去掉开头的 lora_unet_
        if new_key.startswith("lora_unet_"):
            new_key = new_key[len("lora_unet_"):]

        # 去掉结尾的 .lora_up.weight
        for suffix in [".lora_up.weight", ".alpha", ".lora_down.weight"]:
            if new_key.endswith(suffix):
                new_key = new_key[: -len(suffix)]
                break  # 只匹配一个就退出

        # 把 _ 替换成 .
        new_key = new_key.replace("_", ".")

        # 保存到新字典
        cleaned_dict[new_key] = value
    # #根据模型
    # for name, module in root_module.named_modules():
    count = 0
    for key in cleaned_dict.items():
        if "lora.te.text" not in key[0]:
            key_name = key[0]
            for name, module in root_module.named_modules():
                for child_name, child_module in module.named_modules():
                    child_rename = child_name.replace("_", ".")
                    if child_rename == key_name:
                        count += 1
                    else:
                        continue


    return loras

def merge_lora_weights1(
    tensors: torch.Tensor, key: str, prefix: str = "lora_unet_"
) -> Dict[str, torch.Tensor]:
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict. Defaults to "unet.unet.".
    """
    target_key = prefix + key.replace('.', '_')
    out = {}
    for part in ["to_q", "to_k", "to_v", "to_out_0"]:
        down_key = target_key + f"_{part}.lora_down.weight"
        up_key = target_key + f"_{part}.lora_up.weight"
        merged_weight = tensors[up_key] @ tensors[down_key]
        out[part] = merged_weight
    return out


def merge_lora_weights(state_dict, root_module, verbose=True):
    """
    Merges LoRA weights (Up @ Down * scale) for all matching attention blocks in UNet.

    Args:
        state_dict (dict): Loaded LoRA weights from safetensors.
        root_module (torch.nn.Module): The UNet module to match layer names.
        verbose (bool): Print debug info.

    Returns:
        dict: Mapping from module name (e.g. "down_blocks.0.attentions.0...") to merged tensor dict.
              Each value is a dict like {"to_q": merged_tensor, "to_k": ..., "to_v": ..., "to_out.0": ...}
    """

    merged_weights_per_module = {}
    root_module_lora = root_module.replace('.', '_').replace('processor', '')

    for key in state_dict.keys():
        if root_module_lora in key:
            # Extract module key name
            key_name = key.replace("lora_unet_", "").replace(".alpha", "")
            down_key = key.replace(".alpha", ".lora_down.weight")
            up_key = key.replace(".alpha", ".lora_up.weight")

            # Check corresponding weights exist
            if down_key not in state_dict or up_key not in state_dict:
                if verbose:
                    print(f"Missing weights for {key_name}: {down_key} or {up_key}")
                continue

            # Try matching module inside UNet
            matched_module_name = None
            for module_name, module in root_module.named_modules():
                for child_name, child_module in module.named_modules():
                    # Convert "_" back to "." to match naming
                    child_rename = child_name.replace("_", ".")
                    key_rename = key_name.replace("_", ".")
                    if child_rename == key_rename:
                        matched_module_name = f"{module_name}.{child_name}" if module_name else child_name
                        break
                if matched_module_name:
                    break

            if not matched_module_name:
                if verbose:
                    print(f"No module matched for {key_name}")
                continue

            # Merge the weights
            alpha = state_dict[key]
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            else:
                alpha = float(alpha)

            down_weight = state_dict[down_key]
            up_weight = state_dict[up_key]
            rank = down_weight.shape[0]
            scale = alpha / rank

            print(up_weight.shape)
            print(down_weight.shape)

            if up_weight.ndim == 2:
                up_weight = up_weight[:, :, None, None]
            if down_weight.ndim == 2:
                down_weight = down_weight[:, :, None, None]

            merged_tensor = torch.einsum('orhw,rihw->oihw', up_weight, down_weight) * scale

            # 如果多了 (1,1) kernel 维度，去掉
            if merged_tensor.ndim == 4 and merged_tensor.shape[2:] == (1, 1):
                merged_tensor = merged_tensor.squeeze(-1).squeeze(-1)

            # Get suffix type ("to_q", "to_k", etc.)
            suffix = key_name.split(".")[-1]
            if suffix == "to_out_0":
                suffix = "to_out.0"

            if matched_module_name not in merged_weights_per_module:
                merged_weights_per_module[matched_module_name] = {}
            merged_weights_per_module[matched_module_name][suffix] = merged_tensor

            if verbose:
                print(f"Merged {matched_module_name}.{suffix} → shape {merged_tensor.shape}")

    return merged_weights_per_module


def initialize_ziplora_layer(state_dict, state_dict_2, part, **model_kwargs):
    # 在 state_dict 里找包含 part 的完整 key
    matched_keys = [k for k in state_dict.keys() if k.endswith(part)]
    if not matched_keys:
        raise KeyError(f"No matching key found for '{part}' in state_dict")
    if len(matched_keys) > 1:
        print(f"Warning: multiple matches for '{part}', using first: {matched_keys[0]}")

    key = matched_keys[0]

    ziplora_layer = ZipLoRALinearLayer(**model_kwargs)
    ziplora_layer.load_state_dict(
        {
            "weight_1": state_dict[key],
            "weight_2": state_dict_2[key],
        },
        strict=False,
    )
    return ziplora_layer


def initialize_ziplora_layer(state_dict, state_dict_2, part, **model_kwargs):
    ziplora_layer = ZipLoRALinearLayer(**model_kwargs)
    ziplora_layer.load_state_dict(
        {
            "weight_1": state_dict[part],
            "weight_2": state_dict_2[part],
        },
        strict=False,
    )
    return ziplora_layer


def unet_ziplora_state_dict(
    unet: UNet2DConditionModel, quick_release: bool = False
) -> Dict[str, torch.Tensor]:
    r"""
    Returns:
        A state dict containing just the LoRA parameters.
    """
    lora_state_dict = {}

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "get_ziplora_weight"), lora_layer
                weight = lora_layer.get_ziplora_weight()
                lora_state_dict[f"unet.{name}.lora.weight"] = weight

                if quick_release:
                    lora_layer.cpu()
    return lora_state_dict


def ziplora_set_forward_type(unet: UNet2DConditionModel, type: str = "merge"):
    assert type in ["merge", "weight_1", "weight_2"]

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "set_forward_type"), lora_layer
                lora_layer.set_forward_type(type)
    return unet


import torch
import torch.nn as nn
import torch.nn.functional as F
# 确保从你的项目中正确导入 ZipLoRALinearLayer
# (路径可能需要根据你的文件结构调整)
# from .ziplora import ZipLoRALinearLayer # 如果在同一个 utils.py 文件定义，则不需要
# 假设 ZipLoRALinearLayer 已经在当前作用域或被正确导入

def ziplora_compute_mergers_similarity(model: nn.Module) -> torch.Tensor:
    """
    Compute the cosine similarity loss between the merger vectors of all ZipLoRA layers in the model.

    Args:
        model (nn.Module): The model (e.g., UNet) potentially containing ZipLoRALinearLayer instances.

    Returns:
        torch.Tensor: A scalar tensor representing the sum of cosine similarities between
                      merger_1 and merger_2 across all found ZipLoRALinearLayer instances.
                      Returns 0.0 if no such layers with trainable mergers are found.
    """
    similarities = []
    # 遍历模型中的所有模块及其子模块
    for _, module in model.named_modules():
        # 检查模块是否是 ZipLoRALinearLayer 的实例
        if isinstance(module, ZipLoRALinearLayer):
            # 检查该层是否有 merger_1 和 merger_2 属性，并且它们需要梯度（即正在被训练）
            if hasattr(module, 'merger_1') and module.merger_1.requires_grad and \
               hasattr(module, 'merger_2') and module.merger_2.requires_grad:
                # 计算 merger_1 和 merger_2 之间的余弦相似度
                # 添加 epsilon 防止在范数为零时除以零
                sim = F.cosine_similarity(module.merger_1, module.merger_2, dim=0, eps=1e-8)
                similarities.append(sim)

    # 如果没有找到任何可计算相似度的 ZipLoRA 层
    if not similarities:
        # 尝试将 0.0 张量放在模型所在的设备上
        try:
             device = next(model.parameters()).device
             return torch.tensor(0.0, device=device, dtype=torch.float32) # 指定 dtype
        except StopIteration: # 如果模型没有参数
             return torch.tensor(0.0, dtype=torch.float32) # 返回 CPU 上的 0.0


    similarity_sum = torch.stack(similarities).sum(dim=0)

    return similarity_sum


def merge_lora_weights_for_inference(
    tensors: Dict[str, torch.Tensor], key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict. Defaults to "unet.unet.".
    """
    target_key = prefix + key
    out = {}
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        key = target_key + f".{part}.lora.weight"
        out[part] = tensors[key]
    return out


def initialize_ziplora_layer_for_inference(state_dict, part, **model_kwargs):
    ziplora_layer = ZipLoRALinearLayerInference(**model_kwargs)
    ziplora_layer.load_state_dict(
        {
            "weight": state_dict[part],
        },
        strict=False,
    )
    return ziplora_layer


def insert_ziplora_to_unet(
    unet: UNet2DConditionModel, ziplora_name_or_path: str, **kwargs
):
    tensors = get_lora_weights1(ziplora_name_or_path, **kwargs)
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        # Get prepared for ziplora
        attn_name = ".".join(attn_processor_name.split(".")[:-1])
        state_dict = merge_lora_weights_for_inference(tensors, key=attn_name)
        # Set the `lora_layer` attribute of the attention-related matrices.
        kwargs = {"state_dict": state_dict}

        attn_module.to_q.set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                part="to_q",
                **kwargs,
            )
        )
        attn_module.to_k.set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                part="to_k",
                **kwargs,
            )
        )
        attn_module.to_v.set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                part="to_v",
                **kwargs,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                part="to_out.0",
                **kwargs,
            )
        )
    return unet
