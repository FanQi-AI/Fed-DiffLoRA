import os
import torch
import random
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import open_clip
import pandas as pd

# ========== 参数 ==========
lora_dir = "/path/to/dataset/lora"
prompt_file = "/path/to/prompt.txt"
output_dir = "/path/to/images"
result_dir = "/path/to/result"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42

# ========== 设置随机种子 ==========
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# ========== 读取 prompt ==========
with open(prompt_file, 'r', encoding='utf-8') as f:
    prompts = [line.strip() for line in f if line.strip()]

# ========== 加载 CLIP ==========
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model = clip_model.to(device)

# ========== 加载基础 SD1.5 模型 ==========
base_pipe = StableDiffusionPipeline.from_pretrained(
    "/path/to/runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_pipe.scheduler.config)

# ========== 遍历每个 LoRA ==========
all_results = []

lora_files = [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')]

for lora_name in tqdm(lora_files, desc="Processing LoRA files"):
    lora_path = os.path.join(lora_dir, lora_name)
    base_pipe.load_lora_weights(lora_path)

    clip_scores_this_lora = []

    for idx, prompt in enumerate(prompts):
        generator = torch.Generator(device=device).manual_seed(seed)

        # 生成图像
        with torch.autocast(device):
            image = base_pipe(prompt, num_inference_steps=30, guidance_scale=7.5, generator=generator).images[0]

        # 图像预处理 for CLIP
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        text_tensor = clip_tokenizer([prompt]).to(device)

        # 计算 CLIP 相似度
        with torch.no_grad():
            img_feat = clip_model.encode_image(image_tensor)
            txt_feat = clip_model.encode_text(text_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            clip_score = (img_feat @ txt_feat.T).item()

        # 保存中间图片
        image_filename = f"{os.path.splitext(lora_name)[0]}_prompt{idx}.png"
        image.save(os.path.join(output_dir, image_filename))

        result_entry = {
            "lora": lora_name,
            "prompt_idx": idx,
            "prompt": prompt,
            "clip_score": clip_score,
            "image_filename": image_filename
        }
        clip_scores_this_lora.append(result_entry)
        all_results.append(result_entry)

    # 取当前 LoRA 的 top 5
    clip_scores_this_lora.sort(key=lambda x: x["clip_score"], reverse=True)
    for rank, entry in enumerate(clip_scores_this_lora[:5]):
        src_path = os.path.join(output_dir, entry["image_filename"])
        dst_path = os.path.join(result_dir, f"{os.path.splitext(lora_name)[0]}_top{rank+1}.png")
        Image.open(src_path).save(dst_path)

    # 为当前 LoRA 保存 Excel 文件，包含 CLIP 分数和 Prompt 列
    df_lora = pd.DataFrame(clip_scores_this_lora)
    df_lora.to_excel(os.path.join(result_dir, f"{os.path.splitext(lora_name)[0]}_scores_and_prompts.xlsx"), index=False)

    # 保存每个 LoRA 的前五个 prompt 到 txt 文件
    with open(os.path.join(result_dir, f"{os.path.splitext(lora_name)[0]}_top_prompts.txt"), 'w', encoding='utf-8') as f:
        for rank, entry in enumerate(clip_scores_this_lora[:5]):
            f.write(f"Rank {rank+1}: {entry['prompt']}\n")

# ========== 保存所有分数为 Excel ==========
df = pd.DataFrame(all_results)
df.to_excel(os.path.join(result_dir, "all_clip_scores_and_prompts.xlsx"), index=False)

