import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


class Server(object):
    def __init__(self, conf, clients):
        self.conf = conf
        self.clients = clients

        self.embed_model = SentenceTransformer(conf.embed_model_path)

        # 运行期状态
        self.client_payloads: Dict[int, Dict] = {}     # client_id -> {alpha, content_prompt}
        self.clusters: Dict[int, List[int]] = {}        # cluster_id -> [client_id]
        self.cluster_alpha: Dict[int, Dict[str, float]] = {}  # cluster_id -> 聚合 alpha

    # ---------- 1. 收集上传 ----------
    def collect(self):
        self.client_payloads = {}
        for client in self.clients:
            payload = client.send_to_server()
            self.client_payloads[payload["client_id"]] = payload

    # ---------- 2. 内容语义聚类 ----------
    def _get_embeddings(self, prompts: List[str]) -> np.ndarray:
        return np.asarray(self.embed_model.encode(prompts))

    def cluster_by_content(self) -> Dict[int, List[int]]:
        """根据内容 LoRA 语义对客户端做 KMeans 聚类。"""
        client_ids = list(self.client_payloads.keys())
        prompts = [self.client_payloads[cid]["content_prompt"] for cid in client_ids]

        embeddings = self._get_embeddings(prompts)

        num_clusters = self.conf.num_clusters
        if num_clusters is None:
            num_clusters = max(2, int(len(client_ids) ** 0.5))
        num_clusters = min(num_clusters, len(set(prompts)), len(client_ids))
        num_clusters = max(1, num_clusters)

        if num_clusters == 1:
            labels = [0] * len(client_ids)
        else:
            km = KMeans(n_clusters=num_clusters, random_state=self.conf.seed, n_init=10)
            labels = km.fit_predict(embeddings)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for cid, lab in zip(client_ids, labels):
            clusters[int(lab)].append(cid)

        self.clusters = dict(clusters)
        return self.clusters

    # ---------- 3. 逐层 alpha 聚合 ----------
    def aggregate_alpha(self) -> Dict[int, Dict[str, float]]:

        if not self.clusters:
            self.cluster_by_content()

        self.cluster_alpha = {}
        for cluster_id, client_ids in self.clusters.items():
            layer_sum: Dict[str, float] = defaultdict(float)
            layer_cnt: Dict[str, int] = defaultdict(int)

            for cid in client_ids:
                alpha = self.client_payloads[cid]["alpha"]
                for layer_name, val in alpha.items():
                    layer_sum[layer_name] += float(val)
                    layer_cnt[layer_name] += 1

            aggregated = {
                name: layer_sum[name] / layer_cnt[name] for name in layer_sum
            }
            self.cluster_alpha[cluster_id] = aggregated


        return self.cluster_alpha

    # ---------- 4. 下发 ----------
    def distribute(self):

        client_to_cluster = {}
        for cluster_id, client_ids in self.clusters.items():
            for cid in client_ids:
                client_to_cluster[cid] = cluster_id

        id_to_client = {c.client_id: c for c in self.clients}
        for cid, client in id_to_client.items():
            cluster_id = client_to_cluster.get(cid)
            if cluster_id is None:
                continue
            client.receive_from_server(self.cluster_alpha[cluster_id])

