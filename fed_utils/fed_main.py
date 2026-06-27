"""Fed-DiffLoRA 

"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_utils.config import FedConfig, build_client_specs
from fed_utils.client import Client
from fed_utils.server import Server


def parse_args():
    p = argparse.ArgumentParser(description="Fed-DiffLoRA 联邦训练")
    p.add_argument("--clients-data-dir", type=str, default=None,
                   help="客户端数量")
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--embed-model-path", type=str, default=None)
    p.add_argument("--content-lora-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--clusters", type=int, default=None)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--weight-dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()

    conf_kwargs = dict(
        num_rounds=args.rounds,
        num_clusters=args.clusters,
        local_epochs=args.local_epochs,
        device=args.device,
        weight_dtype=args.weight_dtype,
    )
    if args.clients_data_dir:
        conf_kwargs["clients_data_dir"] = args.clients_data_dir
    if args.model_path:
        conf_kwargs["model_path"] = args.model_path
    if args.embed_model_path:
        conf_kwargs["embed_model_path"] = args.embed_model_path
    if args.content_lora_dir:
        conf_kwargs["content_lora_dir"] = args.content_lora_dir
    if args.output_dir:
        conf_kwargs["output_dir"] = args.output_dir

    conf = FedConfig(**conf_kwargs)
    os.makedirs(conf.output_dir, exist_ok=True)


    specs = build_client_specs(conf)
    clients = [Client(conf, spec) for spec in specs]

    server = Server(conf, clients)


    for rnd in range(conf.num_rounds):
        print(f"\n========== 联邦轮次 {rnd + 1}/{conf.num_rounds} ==========")
        server.run_round()

 
    print("\n========== 客户端融合 ==========")
    fusion_summaries = []
    for client in clients:
        save_path = os.path.join(conf.output_dir, f"client_{client.client_id}_fused.pt")
        summary = client.fuse(save_path=save_path)
        fusion_summaries.append(summary)
        print(f"  client {client.client_id}: {summary}")

    report = {
        "num_clients": len(clients),
        "num_clusters": len(server.clusters),
        "clusters": {str(k): v for k, v in server.clusters.items()},
        "cluster_alpha_layer_counts": {
            str(k): len(v) for k, v in server.cluster_alpha.items()
        },
        "fusion_summaries": fusion_summaries,
    }
    report_path = os.path.join(conf.output_dir, "fed_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n========== 完成 ==========")
    print(f"客户端数: {report['num_clients']}  簇数: {report['num_clusters']}")
    print(f"各簇规模: { {k: len(v) for k, v in server.clusters.items()} }")
    print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()