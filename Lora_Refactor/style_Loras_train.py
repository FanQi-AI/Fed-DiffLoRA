import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file, save_file
from transformers import CLIPTokenizer, CLIPTextModel
from Lora import LoRAModule, LoRALayer
from torch import nn, optim
import os
import time  
from typing import List, Tuple, Dict
from PIL import Image
from torchvision import transforms
import os
from collections import defaultdict
import glob
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import torch
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
from transformers import CLIPProcessor, CLIPModel
import pandas as pd

# --- 配置信息 --- 
SD_MODEL_PATH = "/path/to/runwayml/stable-diffusion-v1-5"
LORA_DIR = "/path/to/lora"
DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
WEIGHT_DTYPE = torch.bfloat16  # 或者 torch.float16
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
BATCH_SIZE = 1
PRINT_EVERY = 2
SEED = 42 
image_root_dir = "/path/to/generated_images"
prompts_path="/path/to/prompt.txt"

# -----------------
def encode_image_to_latents(image_path):
    image = Image.open(image_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),  # (C, H, W) in [0, 1]
        transforms.Normalize([0.5], [0.5]),  # to [-1, 1]
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device=device, dtype=weight_dtype)

    # 编码图像为 latent 空间
    with torch.no_grad():
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * 0.18215  # SD 中默认的缩放因子

    return latents

def encode_images_to_latents(image_paths,):
    images = []
    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img)
        images.append(img_tensor)

    image_batch = torch.stack(images).to(device=device, dtype=weight_dtype)

    with torch.no_grad():
        latents = pipe.vae.encode(image_batch).latent_dist.sample()
        latents = latents * 0.18215

    return latents


def build_all_image_paths(image_root_dir):
    all_image_paths = []

    for folder_path in glob.glob(os.path.join(image_root_dir, "*")):
        if os.path.isdir(folder_path):
            lora_name = os.path.basename(folder_path)
            image_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
            for image_path in image_files:
                all_image_paths.append((image_path, lora_name))

    return all_image_paths

def load_image_prompt_pairs(image_dir, prompt_path):
    with open(prompt_path, "r", encoding="gbk") as f:
        prompts = [line.strip() for line in f if line.strip()]

    all_image_paths = sorted([
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    assert len(all_image_paths) == len(prompts), "图像数量和 prompt 数量不一致！"
    return list(zip(all_image_paths, prompts))  # 返回 (image_path, prompt) 的 list

image_prompt_pairs = load_image_prompt_pairs(image_root_dir, prompts_path)

# 加载 Stable Diffusion 模型
# 使用 torch.no_grad() 加载模型以节省内存
with torch.no_grad():
    # 设备 & 数据类型设置
    device = DEVICE
    weight_dtype = WEIGHT_DTYPE
    print(f"使用设备: {device}, 数据类型: {weight_dtype}")
    generator = torch.Generator(device).manual_seed(SEED)

    # --- 加载多个 LoRA 文件 --- (确保这部分正确)
    lora_dir = LORA_DIR
    lora_state_dicts: List[Tuple[str, Dict]] = []  # 显式类型提示
    print(f"从以下目录加载 LoRA 模型: {lora_dir}")
    if os.path.isdir(lora_dir):
        lora_filenames = sorted([f for f in os.listdir(lora_dir) if f.endswith(".safetensors")])
        print(f"  发现 {len(lora_filenames)} 个 .safetensors 文件。")
        for filename in lora_filenames:
            lora_file_path = os.path.join(lora_dir, filename)
            try:
                # 先将 state dict 加载到 CPU，以防 LoRA 过多导致 GPU 显存不足
                state_dict = load_file(lora_file_path, device="cpu")
                # 存储为 (文件名, state_dict) 元组
                lora_state_dicts.append((filename, state_dict))
                print(f"    成功加载 LoRA: {filename}")
            except Exception as e:
                print(f"    加载 LoRA 文件 {filename} 时出错: {e}")
    else:
        print(f"错误: LoRA 目录未找到: {lora_dir}")

    if not lora_state_dicts:
        raise ValueError("没有加载任何 LoRA 文件。请检查 lora_dir 路径并确保其中包含 .safetensors 文件。")
    print(f"已加载 {len(lora_state_dicts)} 个 LoRA 状态字典。")
    # --------------------------------

    # 模型路径
    sd_model_path = SD_MODEL_PATH
    print(f"从以下路径加载基础 Stable Diffusion 模型: {sd_model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(sd_model_path, torch_dtype=weight_dtype).to(device)
    print("基础模型加载完成。")

    # --- 使用所有加载的 LoRA 初始化 LoRAModule --- (传入列表)
    print("初始化 LoRAModule (使用原始结构方法)...")
    # 传入 (文件名, state_dict) 元组的列表
    # 这个 LoRAModule 的 __init__ 方法接收列表，并调用修改后的 *_create_modules
    loramodel = LoRAModule(pipe.text_encoder, pipe.unet, lora_state_dicts)
    # 此处不需要对 loramodel 实例本身调用 .to(device) 或 .to(weight_dtype)
    print("LoRAModule 初始化完成，并通过 apply_to() 应用了 LoRA。")

# --- Wandb 初始化 --- <---- 添加 Wandb 初始化
print("初始化 wandb...")

# print(f"Wandb 初始化完成。项目: {WANDB_PROJECT_NAME}, 运行: {wandb.run.name}")
# ---------------------

# --- 优化器设置 --- (使用 prepare_optimizer_params)
print("设置优化器...")
optimizer_param_groups = loramodel.prepare_optimizer_params()

if not optimizer_param_groups or not optimizer_param_groups[0]["params"]:
    # wandb.log({"error": "No optimizer params found"})
    # wandb.finish()
    raise ValueError(
        "LoRAModule.prepare_optimizer_params() 没有返回任何可优化的参数。请检查 LoRA 加载、命名以及 prepare_optimizer_params 的实现。")

optimizer = optim.Adam(optimizer_param_groups, lr=LEARNING_RATE)
print(f"优化器已为 {len(optimizer_param_groups[0]['params'])} 个 alpha 参数设置完成。")

loss_fn = nn.MSELoss()

# --- 训练参数 ---
num_epochs = NUM_EPOCHS
batch_size = BATCH_SIZE
print_every = PRINT_EVERY

# --- 准备文本输入 ---



# --- Scheduler 设置 ---
num_inference_steps = 50
pipe.scheduler.set_timesteps(num_inference_steps)
timesteps = pipe.scheduler.timesteps.to(device)

# --- 训练设置 ---
pipe.unet.train()
pipe.text_encoder.train()


tokenizer = CLIPTokenizer.from_pretrained(sd_model_path, subfolder="tokenizer")
print("开始训练循环...")
try:

    loss_and_gradients = []
    for epoch in range(num_epochs):
        # --- 创建生成图片保存目录 ---

        epoch_start_time = time.time()
        optimizer.zero_grad()

        # --- 构建图像 + prompt 的 batch ---
        start_idx = (epoch * batch_size) % len(image_prompt_pairs)
        batch_data = image_prompt_pairs[start_idx:start_idx + batch_size]
        batch_paths = [item[0] for item in batch_data]
        batch_prompts = [item[1] for item in batch_data]

        # --- 准备 Latents 和噪声 ---
        latent_height = pipe.unet.config.sample_size
        latent_width = pipe.unet.config.sample_size
        latents_shape = (
            batch_size,
            pipe.unet.config.in_channels,
            latent_height,
            latent_width
        )

        init_latents = encode_images_to_latents(batch_paths)

        t_index = torch.randint(0, len(timesteps), (batch_size,), device=device).long()
        t = timesteps[t_index]
        noise = torch.randn_like(init_latents)
        noisy_latents = pipe.scheduler.add_noise(init_latents, noise, t)

        # --- 获取文本嵌入 ---
        print("对提示词进行分词...")
        text_inputs = tokenizer(
            batch_prompts,  # 传入一整个 prompt batch
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.to(device)
        text_embeddings = pipe.text_encoder(input_ids)[0].to(dtype=weight_dtype)

        # --- 预测噪声 ---
        noise_pred = pipe.unet(noisy_latents, t, encoder_hidden_states=text_embeddings).sample

        # --- 计算损失 ---
        loss = loss_fn(noise_pred, noise)

        # --- 反向传播和优化器步骤 ---
        loss.backward()

        # --- 收集损失和梯度信息 ---
        epoch_data = {"Epoch": epoch, "Loss": loss.item()}
        for name, param in loramodel.named_parameters():
            if param.requires_grad and param.grad is not None:
                epoch_data[name] = param.grad.norm().item()

        # 将当前轮次的数据添加到列表中
        loss_and_gradients.append(epoch_data)

        # --- 输出每个参数的梯度 ---
        # print(f"Epoch {epoch} 梯度统计：")
        # for name, param in loramodel.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f"  - {name}: grad norm = {param.grad.norm().item():.6f}")
        print(f"Epoch {epoch} 梯度统计：")
        total_grad_norm = 0.0
        grad_param_count = 0

        for name, param in loramodel.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                grad_param_count += 1

        if grad_param_count > 0:
            avg_grad_norm = total_grad_norm / grad_param_count
            print(f"平均梯度范数: {avg_grad_norm:.6f}")
        else:
            print("没有可用的梯度参数，无法计算平均梯度范数。")

        optimizer.step()
        if epoch % 5 == 0:
            # 保存 alpha 权重矩阵
            save_dir = f"/path/to/result/line_{epoch}"
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "alpha_values.txt"), "w") as f:
                for name, param in loramodel.named_parameters():
                    if "alpha" in name and param.requires_grad:
                        alpha_val = param.data.to(torch.float32).cpu().numpy().tolist()
                        f.write(f"{name}: {alpha_val}\n")

                
        # --- 日志记录 ---
        if epoch % print_every == 0 or epoch == num_epochs - 1:
            epoch_duration = time.time() - epoch_start_time
            print(f"--- Epoch {epoch}/{num_epochs} --- 损失: {loss.item():.6f} --- 时间: {epoch_duration:.2f}s ---")
            # 打印 alpha 权重以供监控
            print("  当前 Alpha 权重:")

            alpha_values = {}
            # 遍历 LoRAModule 中添加的子模块
            for name, module in loramodel.named_modules():
                # 检查子模块是否为 LoRALayer 实例
                if isinstance(module, LoRALayer):
                    try:
                        # 访问 alpha_model 的权重并获取其值
                        alpha_weight = module.alpha_model.weight.item()
                        # 使用创建时分配的唯一 lora_name
                        # Wandb keys prefer not having complex characters, replace / or \ if needed
                        safe_wandb_key = module.lora_name.replace('/', '_').replace('\\', '_')
                        alpha_values[safe_wandb_key] = alpha_weight  # Use safe key for dict
                    except AttributeError:
                        # 如果 LoRALayer 结构正确，则不应发生这种情况
                        print(f"    警告: 模块 {name} 是 LoRALayer 但缺少 alpha_model 或 weight? 跳过。")

            

finally:
    
    print("结束 wandb 运行...")
   

df = pd.DataFrame(loss_and_gradients)

# 保存到Excel文件
output_file = "/path/to/loss/loss_and_gradients.xlsx"
df.to_excel(output_file, index=False)
print(f"损失和梯度信息已保存到 {output_file}")
print("训练完成。")
