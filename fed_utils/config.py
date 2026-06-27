"""联邦配置
"""
import os
import glob
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---- 默认路径 (按本机现状填写, 可在 FedConfig 中覆盖) ----
DEFAULT_DATASET_ROOT = "/path/to/dataset"
DEFAULT_SD_MODEL_PATH = (
    "runwayml/stable-diffusion-v1-5"
)
DEFAULT_EMBED_MODEL_PATH = os.path.join(
    DEFAULT_DATASET_ROOT, "/path/to/embed_model/all-MiniLM-L6-v2"
)
DEFAULT_CLIENTS_DATA_DIR = os.path.join(DEFAULT_DATASET_ROOT, "clients_data")
DEFAULT_CONTENT_LORA_DIR = os.path.join(DEFAULT_DATASET_ROOT, "content_lora")

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class FedConfig:
    """全局联邦配置"""
    num_rounds: int = 50
    num_clusters: Optional[int] = None  # None -> 自动取 sqrt(N)

    # 路径
    dataset_root: str = DEFAULT_DATASET_ROOT
    model_path: str = DEFAULT_SD_MODEL_PATH
    embed_model_path: str = DEFAULT_EMBED_MODEL_PATH
    clients_data_dir: str = DEFAULT_CLIENTS_DATA_DIR
    content_lora_dir: str = DEFAULT_CONTENT_LORA_DIR
    output_dir: str = "/path/to/result"

    # 本地训练 (alpha) 超参
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 1

    # 设备 / 精度
    device: str = "cuda:0"
    weight_dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    seed: int = 42


@dataclass
class ClientSpec:
    """单个客户端的数据规格 (对应一个 client_X/ 文件夹)。"""
    client_id: int
    client_dir: str                          # client_X/ 路径
    style_lora_files: List[str]              # 该客户端的风格 LoRA 文件 (alpha 训练用)
    content_prompt: str                      # 内容描述 (聚类用)
    image_dir: Optional[str] = None          # 内容图像所在目录
    content_lora_path: Optional[str] = None  # 内容 LoRA (ziplora 融合用, 可选)
    image_paths: List[str] = field(default_factory=list)

    @property
    def style_lora_dir(self) -> str:
        return self.client_dir


def _read_prompt(client_dir: str) -> str:
    """读取客户端内容描述 prompt。"""
    prompt_file = os.path.join(client_dir, "prompt.txt")
    if not os.path.isfile(prompt_file):
        return ""
    for enc in ("utf-8", "gbk"):
        try:
            with open(prompt_file, "r", encoding=enc) as f:
                return f.read().strip()
        except (UnicodeDecodeError, LookupError):
            continue
    return ""


def _find_images(client_dir: str) -> Tuple[Optional[str], List[str]]:
    """查找客户端内容图像。"""
    candidates = [os.path.join(client_dir, "images"), client_dir]
    for d in candidates:
        if not os.path.isdir(d):
            continue
        imgs = sorted(
            p for p in glob.glob(os.path.join(d, "*"))
            if p.lower().endswith(_IMAGE_EXTS)
        )
        if imgs:
            return d, imgs
    return None, []


def _find_content_lora(client_dir: str, style_files: List[str],
                       content_lora_dir: str, client_id: int) -> Optional[str]:
    """确定该客户端的内容 LoRA (ziplora 融合用)。

    """
    # 1. 客户端目录内带 content 标记的 lora
    for f in style_files:
        if "content" in os.path.basename(f).lower():
            return f
    sub = os.path.join(client_dir, "content_lora")
    if os.path.isdir(sub):
        cands = sorted(glob.glob(os.path.join(sub, "*.safetensors")))
        if cands:
            return cands[0]
    # 3. 全局内容 LoRA 池
    if os.path.isdir(content_lora_dir):
        pool = sorted(glob.glob(os.path.join(content_lora_dir, "*.safetensors")))
        if pool:
            return pool[client_id % len(pool)]
    return None


def build_client_specs(conf: FedConfig) -> List[ClientSpec]:
    """扫描 clients_data_dir, 每个 client_X/ 文件夹生成一个 ClientSpec。"""
    root = conf.clients_data_dir
    if not os.path.isdir(root):
        raise FileNotFoundError(f"clients_data 目录不存在: {root}")

    client_dirs = sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    )
    if not client_dirs:
        raise ValueError(f"{root} 下没有任何 client 子目录。")

    specs: List[ClientSpec] = []
    for cid, client_dir in enumerate(client_dirs):
        all_loras = sorted(glob.glob(os.path.join(client_dir, "*.safetensors")))

        content_lora = _find_content_lora(client_dir, all_loras,
                                          conf.content_lora_dir, cid)
    
        style_files = [f for f in all_loras if f != content_lora]
        if not style_files:
            style_files = all_loras

        content_prompt = _read_prompt(client_dir)
        image_dir, image_paths = _find_images(client_dir)

        specs.append(
            ClientSpec(
                client_id=cid,
                client_dir=client_dir,
                style_lora_files=style_files,
                content_prompt=content_prompt,
                image_dir=image_dir,
                content_lora_path=content_lora,
                image_paths=image_paths,
            )
        )

    print(f"[config] 从 {root} 发现 {len(specs)} 个客户端。")
    return specs
