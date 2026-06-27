
import os
import glob
from typing import List, Tuple, Optional

from torch.utils.data import Dataset


class LocalImageDataset(Dataset):
    """单个客户端的本地训练数据 (图像 + prompt)。"""

    def __init__(self, image_dir: Optional[str], prompts_path: Optional[str]):
        self.data: List[Tuple[str, str]] = []

        if not image_dir or not os.path.isdir(image_dir):
            return

        image_paths = sorted(
            p
            for p in glob.glob(os.path.join(image_dir, "*"))
            if p.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        prompts: List[str] = []
        if prompts_path and os.path.isfile(prompts_path):
            for enc in ("utf-8", "gbk"):
                try:
                    with open(prompts_path, "r", encoding=enc) as f:
                        prompts = [ln.strip() for ln in f if ln.strip()]
                    break
                except (UnicodeDecodeError, LookupError):
                    continue

        for i, img in enumerate(image_paths):
            prompt = prompts[i] if i < len(prompts) else (prompts[0] if prompts else "")
            self.data.append((img, prompt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataset(image_dir: Optional[str], prompts_path: Optional[str]) -> LocalImageDataset:
    return LocalImageDataset(image_dir, prompts_path)