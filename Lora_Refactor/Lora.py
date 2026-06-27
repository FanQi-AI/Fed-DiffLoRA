import torch
import torch.nn as nn
from typing import Union, Optional, Literal, List, Tuple, Dict
import types
import functools

def text_encoder_create_modules(root_module, state_dict):
    loras = []
    for key in state_dict:
        if "lora_te" in key and ".alpha" in key:
            key_name_part = key.replace("lora_te_", "").replace(".alpha", "")
            down = key.replace(".alpha", ".lora_down.weight")
            up = key.replace(".alpha", ".lora_up.weight")

            if down not in state_dict or up not in state_dict:
                print(f"    [原始 TE 创建函数] 警告: 键 {key} 缺少对应的 down/up 权重，跳过。")
                continue

            # 遍历模型结构查找目标模块
            target_module_found = False
            key_rename = key_name_part.replace("_", ".")
            for name, module in root_module.named_modules():
                for child_name, child_module in module.named_modules():
                    child_rename = child_name.replace("_", ".")
                    if child_rename == key_rename and isinstance(child_module, (nn.Linear, nn.Conv2d)):
                        try:
                            loralayer = LoRALayer(
                                base_layer=child_module,
                                lora_down=state_dict[down],
                                lora_up=state_dict[up],
                                alpha=state_dict[key],
                                device=root_module.device, # 传递 device 和 dtype
                                weight_dtype=root_module.dtype
                            )
                            loralayer.lora_name = key.replace(".alpha", "") # 原始命名方式
                            loras.append(loralayer)
                            target_module_found = True
                            break # 找到目标，处理下一个 key
                        except Exception as e:
                            print(f"    [原始 TE 创建函数] 错误: 为键 {key} 创建 LoRALayer 时出错: {e}")
                            target_module_found = True # 避免重复警告
                            break
                if target_module_found:
                    break # 找到目标
            

    return loras

def unet_create_modules(root_module, state_dict):
    loras = []
    # 遍历 state_dict 中的键 
    for key in state_dict:
        if "lora_unet" in key and ".alpha" in key:
            key_name_part = key.replace("lora_unet_", "").replace(".alpha", "")
            down = key.replace(".alpha", ".lora_down.weight")
            up = key.replace(".alpha", ".lora_up.weight")

            if down not in state_dict or up not in state_dict:
                print(f"    [原始 UNet 创建函数] 警告: 键 {key} 缺少对应的 down/up 权重，跳过。")
                continue

            # 遍历模型结构查找目标模块 
            target_module_found = False
            key_rename = key_name_part.replace("_", ".")
            for name, module in root_module.named_modules():
                for child_name, child_module in module.named_modules():
                    child_rename = child_name.replace("_", ".")
                    if child_rename == key_rename and isinstance(child_module, (nn.Linear, nn.Conv2d)):
                        try:
                            loralayer = LoRALayer(
                                base_layer=child_module,
                                lora_down=state_dict[down],
                                lora_up=state_dict[up],
                                alpha=state_dict[key],
                                device=root_module.device,
                                weight_dtype=root_module.dtype
                            )
                            loralayer.lora_name = key.replace(".alpha", "")
                            loras.append(loralayer)
                            target_module_found = True
                            break # 找到目标
                        except Exception as e:
                            print(f"    [原始 UNet 创建函数] 错误: 为键 {key} 创建 LoRALayer 时出错: {e}")
                            target_module_found = True
                            break
                if target_module_found:
                    break # 找到目标
          

    return loras

class LoRAModule(nn.Module):
   
    def __init__(self, text_encoder: nn.Module, unet: nn.Module, lora_state_dicts: List[Tuple[str, Dict]]):
        super().__init__()

        self.lora_scale = None # 保留此字段
        print(f"[LoRAModule 初始化] 开始处理 {len(lora_state_dicts)} 个 LoRA 文件...")

        # 初始化总列表
        all_text_encoder_loras: List[LoRALayer] = []
        all_unet_loras: List[LoRALayer] = []

        # 遍历每个 LoRA 文件及其状态字典
        for lora_filename, state_dict in lora_state_dicts:
            print(f"  [LoRAModule 初始化] 处理文件: {lora_filename}")

            # 调用 *原始* 创建函数，处理当前 state_dict
            current_te_loras = text_encoder_create_modules(text_encoder, state_dict)
            current_unet_loras = unet_create_modules(unet, state_dict)
            print(f"    - 为此文件找到 {len(current_te_loras)} 个 TE LoRA 层，{len(current_unet_loras)} 个 UNet LoRA 层。")

            # 为当前文件的 LoRA 层分配唯一的、合法的名称并添加到总列表
            for lora in current_te_loras:
                # 从原始名称 (如 'lora_te_text_model.encoder.layers.0.mlp.fc1') 中提取基础部分
                original_base_name = lora.lora_name.replace("lora_te_", "")
                # 替换文件名和基础名称中的点号为下划线，以创建合法的模块名
                safe_lora_filename = lora_filename.replace('.', '_')
                safe_original_base_name = original_base_name.replace('.', '_')
                unique_safe_name = f"{safe_lora_filename}_te_{safe_original_base_name}"
                lora.lora_name = unique_safe_name # 设置唯一的、合法的名称
                all_text_encoder_loras.append(lora)

            for lora in current_unet_loras:
                original_base_name = lora.lora_name.replace("lora_unet_", "")
                # 替换文件名和基础名称中的点号为下划线
                safe_lora_filename = lora_filename.replace('.', '_')
                safe_original_base_name = original_base_name.replace('.', '_')
                unique_safe_name = f"{safe_lora_filename}_unet_{safe_original_base_name}"
                lora.lora_name = unique_safe_name # 设置唯一的、合法的名称
                all_unet_loras.append(lora)

        # 将包含所有 LoRA 的总列表存储到实例变量中
        self.text_encoder_lora_modules = all_text_encoder_loras
        self.unet_encoder_lora_modules = all_unet_loras


        applied_te_count = 0
        for lora in self.text_encoder_lora_modules: # 确保这里使用的是包含最终 safe name 的 lora 对象
            try:
                lora.apply_to() 
                self.add_module(lora.lora_name, lora)
                applied_te_count += 1
            except Exception as e:
                print(f"  错误: 向文本编码器应用 LoRA 层 {getattr(lora, 'lora_name', 'unknown')} 时出错: {e}")

        applied_unet_count = 0
        for lora in self.unet_encoder_lora_modules: # 确保这里使用的是包含最终 safe name 的 lora 对象
            try:
                lora.apply_to() # 核心：链式替换 forward
                # print(f"  Adding UNet module: {lora.lora_name}") # 调试信息
                self.add_module(lora.lora_name, lora)
                applied_unet_count += 1
            except Exception as e:
                print(f"  错误: 向 UNet 应用 LoRA 层 {getattr(lora, 'lora_name', 'unknown')} 时出错: {e}")
        

    def prepare_optimizer_params(self):
        all_params = []
        print("准备优化器参数...")
        # 从所有 UNet LoRA 层收集 alpha 模型参数
        unet_alpha_params = []
        for lora in self.unet_encoder_lora_modules:
            try:
                unet_alpha_params.extend(list(lora.alpha_model.parameters()))
            except AttributeError:
                 print(f"警告: LoRA 层 {getattr(lora, 'lora_name', 'unknown')} 似乎缺少 alpha_model。")
        print(f"  在 UNet LoRA 中找到 {len(unet_alpha_params)} 个 alpha 参数。")

        # 从所有文本编码器 LoRA 层收集 alpha 模型参数
        te_alpha_params = []
        for lora in self.text_encoder_lora_modules:
             try:
                 te_alpha_params.extend(list(lora.alpha_model.parameters()))
             except AttributeError:
                  print(f"警告: LoRA 层 {getattr(lora, 'lora_name', 'unknown')} 似乎缺少 alpha_model。")
        print(f"  在文本编码器 LoRA 中找到 {len(te_alpha_params)} 个 alpha 参数。")

        all_alpha_params = unet_alpha_params + te_alpha_params

        if not all_alpha_params:
            print("警告: 未找到可优化的 LoRA alpha 参数。")
            return [] # 如果未找到参数，返回空列表

        print(f"找到的总 unique alpha 参数数量: {len(all_alpha_params)}")
        # 以优化器期望的格式返回
        return [{"params": all_alpha_params}]


class LoRALayer(nn.Module):
    # 保持 __init__ 与上个版本一致，确保处理 device/dtype
    def __init__(self, base_layer: nn.Module, lora_down: torch.Tensor, lora_up: torch.Tensor, alpha: torch.Tensor, device: torch.device, weight_dtype = torch.bfloat16):
        super().__init__()
        self.base_layer = base_layer # 保留对基础层的引用
        self.rank = lora_down.shape[0]
        self.org_forward = None # 将由 apply_to 设置
        self.weight_dtype = weight_dtype
        self.device = device # 存储目标设备

        # --- Alpha 设置 --- 确保放置在正确的 device/dtype 上
        # 对非参数张量使用 register_buffer，这些张量应成为 state_dict 的一部分
        self.register_buffer('alpha_val', alpha.squeeze().to(device=self.device, dtype=self.weight_dtype))
        self.alpha_model = nn.Linear(1, 1, bias=False).to(device=self.device, dtype=self.weight_dtype)
        nn.init.constant_(self.alpha_model.weight, 1.0)

        # --- LoRA 权重设置 --- 确保放置在正确的 device/dtype 上
        if isinstance(base_layer, nn.Linear):
            in_features = base_layer.in_features
            out_features = base_layer.out_features
            self.lora_A = nn.Linear(in_features, self.rank, bias=False).to(device=self.device, dtype=self.weight_dtype)
            self.lora_B = nn.Linear(self.rank, out_features, bias=False).to(device=self.device, dtype=self.weight_dtype)
        elif isinstance(base_layer, nn.Conv2d):
            in_channels = base_layer.in_channels
            out_channels = base_layer.out_channels
            # 根据原始代码假设是 1x1 卷积
            self.lora_A = nn.Conv2d(in_channels, self.rank, (1, 1), (1, 1), (0, 0), bias=False).to(device=self.device, dtype=self.weight_dtype)
            self.lora_B = nn.Conv2d(self.rank, out_channels, (1, 1), (1, 1), (0, 0), bias=False).to(device=self.device, dtype=self.weight_dtype)
        else:
            # 像原始代码一样在此处引发错误
            raise ValueError(f"不支持的 LoRA 基础层类型: {type(base_layer)}")

        # 从张量初始化 LoRA 参数，确保它们是 Parameters 并且在正确的 device/dtype 上
        # 以防加载的张量不连续，使用 .contiguous()
        self.lora_A.weight = nn.Parameter(lora_down.clone().contiguous().to(device=self.device, dtype=self.weight_dtype))
        self.lora_B.weight = nn.Parameter(lora_up.clone().contiguous().to(device=self.device, dtype=self.weight_dtype))

        # 此处无需再次调用 self.to()，因为子模块已被显式移动。

    # 保持 apply_to 与原始结构完全一致
    def apply_to(self):
        """ 替换原始 forward 方法 """
        # 备份基础层的 *当前* forward 方法
        self.org_forward = self.base_layer.forward
        # 将基础层的 forward 方法替换为此 LoRA 层的 forward
        self.base_layer.forward = self.forward
        # 保留对 base_layer 的引用，此处不删除

    # 保持 forward 与原始结构完全一致
    def forward(self, x, *args, **kwargs): # 添加 *args, **kwargs 以处理基础 forward 中潜在的额外参数
        # 使用 alpha_model 计算学习到的缩放因子
        # 确保 alpha_val 与模型在同一设备上
        scale = self.alpha_model(self.alpha_val.view(1).to(self.device))

        # 计算 LoRA 路径输出，确保输入 x 在正确的 device/dtype 上
        lora_input = x.to(device=self.device, dtype=self.weight_dtype)
        lora_delta = self.lora_B(self.lora_A(lora_input))

        # 缩放 delta
        scaled_delta = lora_delta * (scale.squeeze() / (self.rank + 1e-9))

        # 调用原始的 forward 方法 (可能是另一个 LoRA 的 forward)
        # 传递任何额外的参数
        original_output = self.org_forward(x, *args, **kwargs)

        # 加上缩放后的 delta，确保 dtypes 匹配
        return original_output + scaled_delta.to(original_output.dtype)
