import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lora_Refactor/Lora.py -> 提供 LoRAModule / LoRALayer
_LORA_REFACTOR_DIR = os.path.join(_REPO_ROOT, "Lora_Refactor")

# Content_style_integration/ziplora_pytorch -> 提供 ZipLoRALinearLayer 等
_ZIPLORA_PARENT_DIR = os.path.join(_REPO_ROOT, "Content_style_integration")


def ensure_on_path():
    for p in (_LORA_REFACTOR_DIR, _ZIPLORA_PARENT_DIR, _REPO_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


ensure_on_path()