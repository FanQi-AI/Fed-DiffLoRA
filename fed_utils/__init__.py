"""Fed-DiffLoRA 联邦学习骨架。

模块组成:
- config.py    : FedConfig 配置 + 100 客户端模拟规格
- dataload.py  : 客户端本地数据集
- models.py    : 基础 SD1.5 加载
- lora_alpha.py: 风格 LoRA 的 alpha 训练 (封装 Lora_Refactor)
- fusion.py    : 内容/风格 ziplora 融合 (封装 ziplora_pytorch)
- client.py    : 联邦客户端 (本地 alpha 训练 -> 上传 -> 接收 -> 融合)
- server.py    : 联邦服务器 (内容语义 KMeans 聚类 -> 逐层 alpha 聚合 -> 下发)
- fed_main.py  : 端到端入口
"""
from .config import FedConfig, ClientSpec, build_client_specs
from .client import Client
from .server import Server

__all__ = [
    "FedConfig",
    "ClientSpec",
    "build_client_specs",
    "Client",
    "Server",
]
