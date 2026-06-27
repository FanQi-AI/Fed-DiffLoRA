import pandas as pd
import argparse
import gc
import itertools
import logging
import math
import os
from diffusers.models.lora import LoRACompatibleLinear

import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr, unet_lora_state_dict
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from ziplora_pytorch import ZipLoRALinearLayer
from ziplora_pytorch.utils import (
    get_lora_weights1,
    get_lora_weights2,
    merge_lora_weights,
    initialize_ziplora_layer,
    unet_ziplora_state_dict,
    ziplora_set_forward_type,
    ziplora_compute_mergers_similarity,
    insert_ziplora_to_unet, merge_lora_weights1,

)
check_min_version("0.24.0.dev0")

logger = get_logger(__name__)
# 参数
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # 预训练模型
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./runwayml/stable-diffusion-v1-5",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models. (e.g., runwayml/stable-diffusion-v1-5)",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. Recommended: stabilityai/sd-vae-ft-mse for SD1.5.",
    )
    # 内容Lora
    parser.add_argument(
        "--lora_name_or_path",
        type=str,
        default="./content/狗写实JeanThePapillion.safetensors",
        required=False,
        help="Path to 1st lora weights",
    )
    # 风格Lora
    parser.add_argument(
        "--lora_name_or_path_2",
        type=str,
        default="/path/to/山水水墨画.safetensors",
        required=False,
        help="Path to 2nd lora weights",
    )
    # alpha矩阵二维
    parser.add_argument(
        "--alpha",
        type=str,
        default="/path/to/alpha_values.txt",
        required=False,
        help="Path to 2nd lora weights",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    # 训练数据地址（图片）
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="/path/to/data",
        help=("A folder containing the training data. "),
    )
    # 训练数据地址（图片）
    parser.add_argument(
        "--instance_data_dir_2",
        type=str,
        default=None,
        help=("A folder containing the training data for 2nd lora"),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat the training data.",
    )
    # prompt1
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default='photo of a sks dog',
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a sks dog', 'in the style of sks'",
    )
    # prompt2
    parser.add_argument(
        "--instance_prompt_2",
        type=str,
        default='in the style of ink',
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a ztz cat', 'in the style of ztz'",
    )
    # 验证提示符
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    # 验证时生成图片数量
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    # 运行dreambooth验证。Dreambooth验证包括运行提示符“”参数
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    # 聚合后保存地址
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/result",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution (default: 512)"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--init_merger_value",
        type=float,
        default=1.0,
        help="initial value of merger coefficient vectors",
    )
    parser.add_argument(
        "--init_merger_value_2",
        type=float,
        default=1.0,
        help="initial value of merger coefficient vectors",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--similarity_lambda",
        type=float,
        default=0.01,
        help="an appropriate multiplier for the cosine similarity loss term",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=1e-03,
        help="Weight decay to use for text_encoder",
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--quick_release",
        action="store_true",
        help="Releases VRAM immediately after processing each layer, conserving it."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError(
            "Specify only one of `--dataset_name` or `--instance_data_dir`"
        )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

# 使用给定的分词器（tokenizer）将文本提示（prompt）转换成模型可以理解的输入格式
def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

# 将文本提示（prompt）通过一个文本编码器（text_encoder）转换成嵌入向量
def encode_prompt(text_encoder, tokenizer, prompt, text_input_ids=None):
    if tokenizer is not None:
        text_input_ids = tokenize_prompt(tokenizer, prompt)

    prompt_embeds_out = text_encoder(
        text_input_ids.to(text_encoder.device),
        output_hidden_states=True,
    )

    prompt_embeds = prompt_embeds_out.last_hidden_state
    return prompt_embeds
# 将多个数据样本（examples）组合成一个批次（batch），以便进行批量处理
def collate_fn(examples):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    pixel_values_2 = [example["instance_images_2"] for example in examples]
    prompts_2 = [example["instance_prompt_2"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values_2 = torch.stack(pixel_values_2)
    pixel_values_2 = pixel_values_2.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "pixel_values_2": pixel_values_2,
        "prompts_2": prompts_2,
    }
    return batch
# 从一个指定的目录中加载图像文件，并将它们转换为 RGB 模式
def prepare_instance_images(instance_data_root: str, repeats: int):
    instance_data_root = Path(instance_data_root)
    if not instance_data_root.exists():
        raise ValueError("Instance images root doesn't exists.")

    instance_images = [
        Image.open(path) for path in list(Path(instance_data_root).iterdir())
    ]

    res = []
    for img in instance_images:
        if img.mode != 'RGB':
             img = img.convert('RGB')
        res.extend(itertools.repeat(img, repeats))
    return res

# 准备和预处理图像数据的 PyTorch 数据集类
class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        instance_data_root_2,
        instance_prompt_2,
        size=512,
        repeats=1,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.instance_prompt_2 = instance_prompt_2

        if args.dataset_name is not None:
            raise NotImplementedError
        self.instance_images = prepare_instance_images(instance_data_root, repeats)
        self.instance_images_2 = prepare_instance_images(instance_data_root_2, repeats)
        self.num_instance_images = max(
            len(self.instance_images), len(self.instance_images_2)
        )
        if len(self.instance_images) < self.num_instance_images:
            self.instance_images = self.instance_images * math.ceil(
                self.num_instance_images / len(self.instance_images)
            )[:self.num_instance_images]

        if len(self.instance_images_2) < self.num_instance_images:
            self.instance_images_2 = self.instance_images_2 * math.ceil(
                self.num_instance_images / len(self.instance_images_2)
            )[:self.num_instance_images]
        self._length = self.num_instance_images
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def _transform_image(self, image):
        image = exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return self.image_transforms(image)

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images[index % len(self.instance_images)]
        example["instance_images"] = self._transform_image(instance_image)
        example["instance_prompt"] = self.instance_prompt
        instance_image_2 = self.instance_images_2[index % len(self.instance_images_2)]
        example["instance_images_2"] = self._transform_image(instance_image_2)
        example["instance_prompt_2"] = self.instance_prompt_2
        return example


# 取文本编码器中所有注意力模块的 LoRA 层的权重，并将它们组织成一个状态字典
def text_encoder_lora_state_dict(text_encoder):
    state_dict = {}

    def text_encoder_attn_modules(text_encoder):
        from transformers import CLIPTextModel

        attn_modules = []

        if isinstance(text_encoder, CLIPTextModel):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))

        return attn_modules

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_layer.{k}"] = v

        for k, v in module.k_proj.lora_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_layer.{k}"] = v

        for k, v in module.v_proj.lora_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_layer.{k}"] = v

        for k, v in module.out_proj.lora_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_layer.{k}"] = v

    return state_dict

# 获取服务器端上传的alpha矩阵
def parse_lora_alpha_file(file_path):
    """
    从给定txt文件解析 Lora alpha 值并生成 DataFrame。
    行：层名，列：Lora名，值：alpha（无值填1）

    :param file_path: str, txt文件路径
    :return: pandas.DataFrame
    """
    data = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 分割字符串：Lora名_safetensors_层名: [[alpha值]]
            lora_full, value_part = line.split(':')
            alpha_value = float(value_part.strip().lstrip('[').lstrip('[').rstrip(']').rstrip(']'))

            # 拆出 Lora 名和 层名
            if '_safetensors_' in lora_full:
                lora_name, layer_name = lora_full.split('_safetensors_', 1)
            else:
                continue  # 跳过不匹配的行

            # 填入数据
            if layer_name not in data:
                data[layer_name] = {}
            data[layer_name][lora_name] = alpha_value

    # 提取所有唯一 Lora 名和层名
    all_loras = sorted({lora for layer in data.values() for lora in layer.keys()})
    all_layers = sorted(data.keys())

    # 构建 DataFrame，默认填1
    df = pd.DataFrame(1, index=all_layers, columns=all_loras)

    # 填入实际的 alpha 值
    for layer, lora_values in data.items():
        for lora, alpha in lora_values.items():
            df.at[layer, lora] = alpha

    return df


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.train_text_encoder)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name,
            exist_ok=True,
            token=args.hub_token,
        ).repo_id

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    text_encoder_cls = CLIPTextModel

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
    )
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)

    vae.to(accelerator.device, dtype=torch.float32)

    text_encoder.to(accelerator.device, dtype=weight_dtype)

    lora_weights = get_lora_weights1(args.lora_name_or_path)
    lora_weights_2 = get_lora_weights1(args.lora_name_or_path_2)
    # lora_weights_2 = get_lora_weights2(args.lora_name_or_path_2)


    unet_lora_parameters = []
    logger.info("Initializing ZipLoRA layers by iterating through attention processors...")

    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # 解析 attention 模块
        attn_module = unet
        try:
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)
        except AttributeError:
            logger.warning(f"Skipping {attn_processor_name} as it was not found in UNet.")
            continue

        attn_name = ".".join(attn_processor_name.split(".")[:-1])
        merged_lora_weights_dict = merge_lora_weights1(lora_weights, attn_name)
        merged_lora_weights_dict_2 = merge_lora_weights1(lora_weights_2, attn_name)
        kwargs = {
            "state_dict": merged_lora_weights_dict,
            "state_dict_2": merged_lora_weights_dict_2,
        }

        def patch_unet_with_lora(unet):
            for module_name, module in unet.named_modules():
                for child_name, child_module in module.named_children():
                    # 确保 child_module 是 nn.Linear，但不是 LoRACompatibleLinear
                    if isinstance(child_module, nn.Linear) and not isinstance(child_module, LoRACompatibleLinear):
                        print(f"Replacing {module_name}.{child_name}: nn.Linear → LoRACompatibleLinear")

                        # 创建新的 LoRACompatibleLinear 层
                        new_layer = LoRACompatibleLinear(
                            child_module.in_features,
                            child_module.out_features,
                            bias=child_module.bias is not None
                        )

                        # 复制原始权重和偏置
                        new_layer.weight.data = child_module.weight.data.clone()
                        if child_module.bias is not None:
                            new_layer.bias.data = child_module.bias.data.clone()

                        # 替换模块
                        setattr(module, child_name, new_layer)

        patch_unet_with_lora(pipe.unet)

        print(type(attn_module.to_q))
        # Set the `lora_layer` attribute of the attention-related matrices.
        # attn_module.to_q = LoRACompatibleLinear(
        #     attn_module.to_q.in_features,
        #     attn_module.to_q.out_features,
        # )
        attn_module.to_q.set_lora_layer(
            initialize_ziplora_layer(
                part="to_q",
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                init_merger_value=args.init_merger_value,
                init_merger_value_2=args.init_merger_value_2,
                **kwargs,
            )
        )
        attn_module.to_k.set_lora_layer(
            initialize_ziplora_layer(
                part="to_k",
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                init_merger_value=args.init_merger_value,
                init_merger_value_2=args.init_merger_value_2,
                **kwargs,
            )
        )
        attn_module.to_v.set_lora_layer(
            initialize_ziplora_layer(
                part="to_v",
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                init_merger_value=args.init_merger_value,
                init_merger_value_2=args.init_merger_value_2,
                **kwargs,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            initialize_ziplora_layer(
                part="to_out.0",
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                init_merger_value=args.init_merger_value,
                init_merger_value_2=args.init_merger_value_2,
                **kwargs,
            )
        )

        # Accumulate the LoRA params to optimize.
        unet_lora_parameters.extend(
            [p for p in attn_module.to_q.lora_layer.parameters() if p.requires_grad]
        )
        unet_lora_parameters.extend(
            [p for p in attn_module.to_k.lora_layer.parameters() if p.requires_grad]
        )
        unet_lora_parameters.extend(
            [p for p in attn_module.to_v.lora_layer.parameters() if p.requires_grad]
        )
        unet_lora_parameters.extend(
            [
                p
                for p in attn_module.to_out[0].lora_layer.parameters()
                if p.requires_grad
            ]
        )

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        raise NotImplementedError
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(
            text_encoder_one, dtype=torch.float32, rank=args.rank
        )
        text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(
            text_encoder_two, dtype=torch.float32, rank=args.rank
        )

        #
        # # 合并 LoRA 权重
        # # merged_lora_weights_dict = merge_lora_weights(lora_weights, unet)
        # # merged_lora_weights_dict_2 = merge_lora_weights(lora_weights_2, unet)
        #
        # for part_key, merged_weight in merged_lora_weights_dict.items():
        #     try:
        #         sub_module = getattr(attn_module, part_key)
        #         if hasattr(sub_module, 'weight'):
        #             logger.info(f"Applying merged LoRA weight to {attn_name}.{part_key}")
        #             sub_module.weight.data += merged_weight.to(sub_module.weight.device)
        #             unet_lora_parameters.append(sub_module.weight)
        #         else:
        #             logger.warning(f"{attn_name}.{part_key} has no .weight attribute")
        #     except AttributeError:
        #         logger.warning(f"{attn_name}.{part_key} not found in {attn_module}")
        #
        # # *** Add check: Skip this attn block if merged weights are missing for either LoRA ***
        # if not merged_lora_weights_dict or not merged_lora_weights_dict_2:
        #     logger.warning(f"Skipping ZipLoRA initialization for {attn_name} due to missing merged weights in one or both base LoRAs.")
        #     continue # Skip to the next attention processor
        #
        # kwargs = {
        #     "state_dict": merged_lora_weights_dict,
        #     "state_dict_2": merged_lora_weights_dict_2,
        # }
        #
        # # Set the `lora_layer` attribute of the attention-related matrices.
        # if hasattr(attn_module, "to_q") and attn_module.to_q is not None:
        #     # Get features before potential replacement if needed later
        #     in_features = attn_module.to_q.in_features
        #     out_features = attn_module.to_q.out_features
        #     # Initialize the ZipLoRA layer
        #     # ziplora_q_layer = initialize_ziplora_layer(
        #     #     part="to_q",
        #     #     in_features=in_features,
        #     #     out_features=out_features,
        #     #     init_merger_value=args.init_merger_value,
        #     #     init_merger_value_2=args.init_merger_value_2,
        #     #     **kwargs,
        #     # )
        #     merged_lora_weights_dict = {k: list(v.values())[0] for k, v in merged_lora_weights_dict.items()}
        #     merged_lora_weights_dict_2 = {k: list(v.values())[0] for k, v in merged_lora_weights_dict_2.items()}
        #
        #     ziplora_q_layer = initialize_ziplora_layer(
        #         part="to_q",
        #         in_features=in_features,
        #         out_features=out_features,
        #         init_merger_value=args.init_merger_value,
        #         init_merger_value_2=args.init_merger_value_2,
        #         state_dict=merged_lora_weights_dict,
        #         state_dict_2=merged_lora_weights_dict_2,
        #     )
        #
        #     # Direct replacement
        #     attn_module.to_q = ziplora_q_layer
        #     # Collect parameters from the new layer
        #     unet_lora_parameters.extend([p for p in ziplora_q_layer.parameters() if p.requires_grad])
        #
        # if hasattr(attn_module, "to_k") and attn_module.to_k is not None:
        #     in_features = attn_module.to_k.in_features
        #     out_features = attn_module.to_k.out_features
        #     ziplora_k_layer = initialize_ziplora_layer(
        #         part="to_k",
        #         in_features=in_features,
        #         out_features=out_features,
        #         init_merger_value=args.init_merger_value,
        #         init_merger_value_2=args.init_merger_value_2,
        #         state_dict={k: v for k, v in merged_lora_weights_dict.items() if 'to_k' in k},
        #         state_dict_2={k: v for k, v in merged_lora_weights_dict_2.items() if 'to_k' in k},
        #     )
        #     attn_module.to_k = ziplora_k_layer
        #     unet_lora_parameters.extend([p for p in ziplora_k_layer.parameters() if p.requires_grad])
        #
        # if hasattr(attn_module, "to_v") and attn_module.to_v is not None:
        #     in_features = attn_module.to_v.in_features
        #     out_features = attn_module.to_v.out_features
        #     ziplora_v_layer = initialize_ziplora_layer(
        #         part="to_v",
        #         in_features=in_features,
        #         out_features=out_features,
        #         init_merger_value=args.init_merger_value,
        #         init_merger_value_2=args.init_merger_value_2,
        #         state_dict={k: v for k, v in merged_lora_weights_dict.items() if 'to_v' in k},
        #         state_dict_2={k: v for k, v in merged_lora_weights_dict_2.items() if 'to_v' in k},
        #     )
        #     attn_module.to_v = ziplora_v_layer
        #     unet_lora_parameters.extend([p for p in ziplora_v_layer.parameters() if p.requires_grad])
        #
        # if hasattr(attn_module, "to_out") and len(attn_module.to_out) > 0 and attn_module.to_out[0] is not None:
        #      # Check if it's a Linear layer before replacing
        #      if isinstance(attn_module.to_out[0], torch.nn.Linear):
        #          in_features = attn_module.to_out[0].in_features
        #          out_features = attn_module.to_out[0].out_features
        #          ziplora_out_layer = initialize_ziplora_layer(
        #              part="to_out.0",
        #              in_features=in_features,
        #              out_features=out_features,
        #              init_merger_value=args.init_merger_value,
        #              init_merger_value_2=args.init_merger_value_2,
        #              state_dict={k: v for k, v in merged_lora_weights_dict.items() if 'to_out.0' in k},
        #              state_dict_2={k: v for k, v in merged_lora_weights_dict_2.items() if 'to_out.0' in k},
        #          )
        #          attn_module.to_out[0] = ziplora_out_layer
        #          unet_lora_parameters.extend([p for p in ziplora_out_layer.parameters() if p.requires_grad])
        #      else:
        #           logger.warning(f"Skipping ZipLoRA for {attn_name}.to_out.0 as it's not a Linear layer ({type(attn_module.to_out[0])})")

    text_lora_parameters = None
    if args.train_text_encoder:
        if not hasattr(args, 'rank'):
             args.rank = 4
             logger.warning(f"LoRA rank not specified, using default rank={args.rank} for text encoder.")

        text_lora_parameters = LoraLoaderMixin._modify_text_encoder(
            text_encoder, dtype=torch.float32, rank=args.rank
        )
        logger.info(f"Added LoRA layers to Text Encoder with rank {args.rank}.")

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None
            text_encoder_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = unet_ziplora_state_dict(model)
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                    if args.train_text_encoder:
                        text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(
                            model
                        )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                weights.pop()

            unet_state_dict_to_save = {}
            if unet_lora_layers_to_save:
                unet_state_dict_to_save = unet_lora_layers_to_save

            text_encoder_state_dict_to_save = {}
            if text_encoder_lora_layers_to_save:
                text_encoder_state_dict_to_save = text_encoder_lora_layers_to_save

            StableDiffusionPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_state_dict_to_save,
                text_encoder_lora_layers=text_encoder_state_dict_to_save,
            )

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    unet_lora_parameters_with_lr = {
        "params": unet_lora_parameters,
        "lr": args.learning_rate,
    }
    total_params_unet = sum(p.numel() for p in unet_lora_parameters if p.requires_grad)
    logger.info(f"Number of Trainable UNet Parameters: {total_params_unet * 1.e-6:.2f} M")

    params_to_optimize = []
    unet_lora_parameters_with_lr = {
        "params": unet_lora_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize.append(unet_lora_parameters_with_lr)

    if args.train_text_encoder and text_lora_parameters is not None:
        text_lora_parameters_with_lr = {
            "params": text_lora_parameters,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize.append(text_lora_parameters_with_lr)
        total_params_te = sum(p.numel() for p in text_lora_parameters if p.requires_grad)
        logger.info(f"Number of Trainable Text Encoder Parameters: {total_params_te * 1.e-6:.2f} M")
        total_params = total_params_unet + total_params_te
        logger.info(f"Total Number of Trainable Parameters: {total_params * 1.e-6:.2f} M")
    else:
        logger.info(f"Total Number of Trainable Parameters: {total_params_unet * 1.e-6:.2f} M")

    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warn(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warn(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warn(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr and len(params_to_optimize) > 1:
            logger.warn(
                f"Learning rates were provided both for the unet and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate for all groups."
            )
            params_to_optimize[1]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        instance_data_root_2=args.instance_data_dir_2,
        instance_prompt_2=args.instance_prompt_2,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    instance_prompt_hidden_states = None
    instance_prompt_hidden_states_2 = None
    if not args.train_text_encoder:
        instance_prompt_hidden_states = encode_prompt(
            text_encoder, tokenizer, args.instance_prompt
        )
        instance_prompt_hidden_states_2 = encode_prompt(
            text_encoder, tokenizer, args.instance_prompt_2
        )

    gc.collect()
    torch.cuda.empty_cache()

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if args.train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth-ziplora-sd15", config=vars(args))

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(args.resume_from_checkpoint.split("-")[-1])
        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()

            text_encoder.text_model.embeddings.requires_grad_(True)
            try:
                 text_encoder.text_model.embeddings.requires_grad_(True)
            except AttributeError:
                 logger.warning("Could not set requires_grad on text_encoder embeddings for gradient checkpointing.")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Get prompt embeddings
                if args.train_text_encoder:
                    # Tokenize and encode prompts within the loop
                    # SD1.5: Need the prepared text_encoder and tokenizer here
                    current_text_encoder = accelerator.unwrap_model(text_encoder) if args.train_text_encoder else text_encoder
                    prompt_embeds = encode_prompt(current_text_encoder, tokenizer, batch["prompts"])
                    prompt_embeds_2 = encode_prompt(current_text_encoder, tokenizer, batch["prompts_2"])
                else:
                    # Use pre-computed embeddings
                    prompt_embeds = instance_prompt_hidden_states.repeat(len(batch["prompts"]), 1, 1)
                    prompt_embeds_2 = instance_prompt_hidden_states_2.repeat(len(batch["prompts_2"]), 1, 1)

                model_inputs = []
                pixel_values_list = [batch["pixel_values"], batch["pixel_values_2"]]

                for i in range(2):
                    pixel_values = pixel_values_list[i].to(dtype=vae.dtype)
                    with torch.no_grad():
                        model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                    if args.pretrained_vae_model_name_or_path is None:
                        model_input = model_input.to(dtype=weight_dtype)
                    model_inputs.append(model_input)

                noise = torch.randn_like(model_inputs[0])
                bsz = model_inputs[0].shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=model_inputs[0].device,
                )
                timesteps = timesteps.long()

                noisy_latents = []
                for i in range(2):
                    noisy_latents.append(noise_scheduler.add_noise(
                        model_inputs[i], noise, timesteps
                    ))


                # 1. merged weights + concept
                ziplora_set_forward_type(unet, type="merge")
                model_pred_mc = unet(noisy_latents[0], timesteps, encoder_hidden_states=prompt_embeds).sample

                # 2. merged weights + style
                model_pred_ms = unet(noisy_latents[1], timesteps, encoder_hidden_states=prompt_embeds_2).sample

                # 3. concept weights + concept (Re-insert this block)
                ziplora_set_forward_type(unet, type="weight_1")
                with torch.no_grad():
                    model_pred_cc = unet(noisy_latents[0], timesteps, encoder_hidden_states=prompt_embeds).sample

                # 4. style weights + style (Re-insert this block)
                ziplora_set_forward_type(unet, type="weight_2")
                with torch.no_grad():
                    model_pred_ss = unet(noisy_latents[1], timesteps, encoder_hidden_states=prompt_embeds_2).sample


                # compute losses
                ziplora_set_forward_type(unet, type="merge")

                if args.snr_gamma is None:
                    loss_1 = F.mse_loss(
                        model_pred_mc.float(), model_pred_cc.float(), reduction="mean"
                    )
                    loss_2 = F.mse_loss(
                        model_pred_ms.float(), model_pred_ss.float(), reduction="mean"
                    )
                    loss_3 = args.similarity_lambda * ziplora_compute_mergers_similarity(
                        unet
                    )
                    loss = loss_1 + loss_2 + loss_3
                else:
                    loss = loss_1 + loss_2

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet_lora_parameters
                    if args.train_text_encoder:
                         params_to_clip = itertools.chain(unet_lora_parameters, text_lora_parameters)
                    accelerator.clip_grad_norm_(
                        params_to_clip, args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "loss_1": loss_1.detach().item(),
                "loss_2": loss_2.detach().item(),
                "loss_3": loss_3.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if accelerator.is_main_process:
             if (
                 args.validation_prompt is not None
                 and epoch % args.validation_epochs == 0
             ):
                 logger.info(
                     f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                     f" {args.validation_prompt}."
                 )
                 pipeline = StableDiffusionPipeline.from_pretrained(
                     args.pretrained_model_name_or_path,
                     vae=accelerator.unwrap_model(vae),
                     text_encoder=accelerator.unwrap_model(text_encoder),
                     unet=accelerator.unwrap_model(unet),
                     revision=args.revision,
                     torch_dtype=weight_dtype,
                 )

                 # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                 scheduler_args = {} # Define scheduler_args as empty dict

                 pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                     pipeline.scheduler.config, **scheduler_args
                 )

                 pipeline = pipeline.to(accelerator.device)
                 pipeline.set_progress_bar_config(disable=True)

                 generator = (
                     torch.Generator(device=accelerator.device).manual_seed(args.seed)
                     if args.seed
                     else None
                 )
                 pipeline_args = {"prompt": args.validation_prompt}

                 with torch.no_grad():
                    images = [
                        pipeline(**pipeline_args, generator=generator, num_inference_steps=25).images[0]
                        for _ in range(args.num_validation_images)
                    ]

                 for tracker in accelerator.trackers:
                     if tracker.name == "tensorboard":
                         np_images = np.stack([np.asarray(img) for img in images])
                         tracker.writer.add_images(
                             "validation", np_images, epoch, dataformats="NHWC"
                         )
                     if tracker.name == "wandb":
                         tracker.log(
                             {
                                 "validation": [
                                     wandb.Image(
                                         image, caption=f"{i}: {args.validation_prompt}"
                                     )
                                     for i, image in enumerate(images)
                                 ]
                             }
                         )

                 del pipeline
                 torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = unet_ziplora_state_dict(unet, args.quick_release)

        text_encoder_lora_layers = None
        if args.train_text_encoder:
            text_encoder = accelerator.unwrap_model(text_encoder)
            text_encoder_lora_layers = text_encoder_lora_state_dict(
                text_encoder.to(torch.float32)
            )

        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
        )

        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )

        # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
        scheduler_args = {} # Define scheduler_args as empty dict

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config, **scheduler_args
        )

        pipeline.unet = insert_ziplora_to_unet(pipeline.unet, args.output_dir, is_final_weights=True)
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
            generator = (
                torch.Generator(device=accelerator.device).manual_seed(args.seed)
                if args.seed
                else None
            )
            images = [
                pipeline(
                    args.validation_prompt, num_inference_steps=25, generator=generator
                ).images[0]
                for _ in range(args.num_validation_images)
            ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        "test", np_images, epoch, dataformats="NHWC"
                    )
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(
                                    image, caption=f"{i}: {args.validation_prompt}"
                                )
                                for i, image in enumerate(images)
                            ]
                        }
                    )

    accelerator.end_training()


if __name__ == "__main__":

    args = parse_args()
    main(args)
