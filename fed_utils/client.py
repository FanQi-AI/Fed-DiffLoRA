"""联邦客户端
  1. local_train()    : 用 Lora_Refactor 训练本地风格 LoRA 的 alpha 系数
  2. send_to_server() : 上传 
  3. receive_from_server(): 接收服务器下发的、本簇聚合后的 alpha
  4. fuse()           : 用 ziplora 融合内容 LoRA + 风格 LoRA

"""
from typing import Dict, Optional

from .dataload import get_dataset
from .lora_alpha import train_style_alphas
from .fusion import fuse_content_style


class Client(object):
    def __init__(self, conf, spec):
        """
        - conf: FedConfig
        - spec: ClientSpec (该客户端的风格 LoRA / 内容 LoRA / 内容 prompt 等)
        """
        self.conf = conf
        self.spec = spec
        self.client_id = spec.client_id

        # 本地训练数据 
        self.train_dataset = get_dataset(spec.image_dir, spec.prompts_path)

        # 训练得到的本地 alpha; 服务器下发的聚合 alpha
        self.local_alpha: Dict[str, float] = {}
        self.aggregated_alpha: Dict[str, float] = {}

    def local_train(self) -> Dict[str, float]:
        """本地训练风格 LoRA 的 alpha 系数。"""
        self.local_alpha = train_style_alphas(self.conf, self.spec)
        return self.local_alpha

    def send_to_server(self) -> Dict:
        """上传 alpha 与内容语义 prompt。"""
        if not self.local_alpha:
            self.local_train()
        return {
            "client_id": self.client_id,
            "alpha": self.local_alpha,
            "content_prompt": self.spec.content_prompt,
        }

    def receive_from_server(self, aggregated_alpha: Dict[str, float]):
        """接收本簇聚合后的 alpha。"""
        self.aggregated_alpha = dict(aggregated_alpha)
        print(
            f"客户端 {self.client_id}: 收到聚合 alpha "
            f"({len(self.aggregated_alpha)} 层)。"
        )

    def fuse(self, save_path: Optional[str] = None) -> Dict:
        """用 ziplora 融合内容 LoRA 与 (带聚合 alpha 的) 风格 LoRA。"""
        alpha = self.aggregated_alpha or self.local_alpha
        summary = fuse_content_style(self.conf, self.spec, alpha, save_path=save_path)
        return summary