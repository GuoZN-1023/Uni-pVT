# train.py
import os
import argparse
import yaml
import torch

from models.fusion_model import FusionModel
from utils.dataset import get_dataloaders
from utils.physics_loss import PhysicsLoss
from utils.trainer import train_model
from utils.logger import get_file_logger


def _bridge_loss_cfg_to_training(cfg: dict):
    """
    兼容旧配置：你的 loss 超参在 cfg["loss"]，而新版 PhysicsLoss(cfg) 默认从 cfg["training"] 里读。
    这里做一个“桥接”，不要求你重写 config。
    """
    cfg.setdefault("training", {})
    loss_cfg = cfg.get("loss", {}) or {}
    tr = cfg["training"]

    # 把旧 loss 字段映射到 training（如果 training 里已经显式写了，就不覆盖）
    mapping = {
        "lambda_nonneg": "lambda_nonneg",
        "lambda_smooth": "lambda_smooth",
        "lambda_entropy": "lambda_entropy",
        "loss_type": "loss_type",
        "huber_delta": "huber_delta",
        # 下面这些如果你旧版 PhysicsLoss 用到了，但新版没用也没关系；放着不影响
        "lambda_extreme": "lambda_extreme",
        "lambda_relative": "lambda_relative",
        "extreme_alpha": "extreme_alpha",
    }

    for k_loss, k_tr in mapping.items():
        if k_tr not in tr and k_loss in loss_cfg:
            tr[k_tr] = loss_cfg[k_loss]

    # region_weights 如果你放在 cfg["loss"] 里，也桥接过去；一般建议你放在 training
    if "region_weights" not in tr and "region_weights" in loss_cfg:
        tr["region_weights"] = loss_cfg["region_weights"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="可选：覆盖 config 中的 data 路径，直接指定某个 csv 文件",
    )
    args = parser.parse_args()

    # 读取配置
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 如果命令行指定了 data，覆盖配置中的路径
    if args.data is not None:
        cfg.setdefault("paths", {})
        cfg["paths"]["data"] = args.data

    # 对齐 loss 配置（关键）
    _bridge_loss_cfg_to_training(cfg)

    # 路径准备
    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)

    cfg["paths"]["scaler"] = cfg["paths"].get("scaler", os.path.join(save_dir, "scaler.pkl"))

    # logger
    log_file = os.path.join(save_dir, "logs", "training.log")
    logger = get_file_logger(log_file, name="train")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # 构建 DataLoader
    dataloaders = get_dataloaders(cfg)

    # 从数据自动检测 input_dim，并写回 cfg["model"]["input_dim"]
    train_loader = dataloaders["train"]
    if hasattr(train_loader.dataset, "dataset"):
        full_dataset = train_loader.dataset.dataset
    else:
        full_dataset = train_loader.dataset

    real_input_dim = int(full_dataset.input_dim)
    cfg.setdefault("model", {})
    cfg["model"]["input_dim"] = real_input_dim
    logger.info(f"Detected input_dim={real_input_dim}, set cfg['model']['input_dim'] accordingly.")

    # 记录本次实际使用的配置（注意：检测到 input_dim、桥接后再写）
    with open(os.path.join(save_dir, "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    # 构建模型
    model = FusionModel(cfg).to(device)

    # 构建损失函数（新版：PhysicsLoss(cfg)）
    criterion = PhysicsLoss(cfg)

    # 优化器（trainer 里会按阶段自己建 gate/expert/joint optimizer；这里保留接口不破坏调用）
    lr = float(cfg["training"]["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练
    train_model(model, dataloaders, criterion, optimizer, cfg, device, logger)


if __name__ == "__main__":
    main()
