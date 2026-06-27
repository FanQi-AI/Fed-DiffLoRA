import os
import ast
import argparse
from typing import List, Tuple, Dict
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

from Lora import LoRAModule


def parse_args():
    parser = argparse.ArgumentParser(description="style LoRA + content LoRA")
    parser.add_argument("--style_lora_dir", type=str, required=True)
    parser.add_argument("--content_lora", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./infer_out")
    parser.add_argument("--sd_model_path", type=str, default="/path/to/runwayml/stable-diffusion-v1-5")
    parser.add_argument("--alpha_file", type=str, default=None)
    parser.add_argument("--content_weight", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_style_lora_state_dicts(style_lora_dir: str) -> List[Tuple[str, Dict]]:
    if not os.path.isdir(style_lora_dir):
        raise FileNotFoundError(f"风格 LoRA 目录不存在: {style_lora_dir}")

    lora_filenames = sorted(f for f in os.listdir(style_lora_dir) if f.endswith(".safetensors"))
    if not lora_filenames:
        raise ValueError(f"目录中没有 .safetensors 文件: {style_lora_dir}")

    lora_state_dicts: List[Tuple[str, Dict]] = []
    for filename in lora_filenames:
        path = os.path.join(style_lora_dir, filename)
        lora_state_dicts.append((filename, load_file(path, device="cpu")))
    return lora_state_dicts


def load_trained_alphas(loramodel: LoRAModule, alpha_file: str):
    if not os.path.isfile(alpha_file):
        return

    name_to_value: Dict[str, float] = {}
    with open(alpha_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ": " not in line:
                continue
            name, value_str = line.split(": ", 1)
            try:
                value = ast.literal_eval(value_str)
                while isinstance(value, (list, tuple)):
                    value = value[0]
                name_to_value[name.strip()] = float(value)
            except Exception:
                continue

    state = dict(loramodel.named_parameters())
    for name, value in name_to_value.items():
        if name in state:
            with torch.no_grad():
                state[name].fill_(value)


def main():
    args = parse_args()
    device = args.device
    weight_dtype = get_dtype(args.dtype)
    os.makedirs(args.output_dir, exist_ok=True)

    style_lora_state_dicts = load_style_lora_state_dicts(args.style_lora_dir)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_path, torch_dtype=weight_dtype
    ).to(device)

    loramodel = LoRAModule(pipe.text_encoder, pipe.unet, style_lora_state_dicts)
    if args.alpha_file:
        load_trained_alphas(loramodel, args.alpha_file)

    pipe.unet.eval()
    pipe.text_encoder.eval()

    use_content = args.content_lora is not None
    if use_content:
        if not os.path.isfile(args.content_lora):
            raise FileNotFoundError(f"内容 LoRA 不存在: {args.content_lora}")
        pipe.load_lora_weights(args.content_lora, adapter_name="lora_extra")
        pipe.set_adapters(["lora_extra"], adapter_weights=[args.content_weight])
        content_name = os.path.splitext(os.path.basename(args.content_lora))[0]
        weight_tag = f"cw{args.content_weight}"
    else:
        content_name = "no_content"
        weight_tag = "style_only"

    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        image = pipe(
            args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

    out_path = os.path.join(args.output_dir, f"{content_name}_{weight_tag}_seed{args.seed}.png")
    image.save(out_path)
    print(out_path)

    if use_content:
        pipe.unload_lora_weights()


if __name__ == "__main__":
    main()
