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
        cfg["paths"]["data"] = args.data

    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    cfg["paths"]["scaler"] = cfg["paths"].get(
        "scaler", os.path.join(save_dir, "scaler.pkl")
    )

    # ==== 提前创建 logger，这样后面任何地方都可以用 ====
    log_file = os.path.join(save_dir, "logs", "training.log")
    logger = get_file_logger(log_file, name="train")

    # 记录本次实际使用的配置
    with open(os.path.join(save_dir, "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # 构建 DataLoader
    dataloaders = get_dataloaders(cfg)

    # ==== 从数据自动检测 input_dim，并写回 cfg["model"]["input_dim"] ====
    train_loader = dataloaders["train"]
    # random_split 返回的是 Subset，真正的 ZDataset 在 .dataset 里
    if hasattr(train_loader.dataset, "dataset"):
        full_dataset = train_loader.dataset.dataset
    else:
        full_dataset = train_loader.dataset

    real_input_dim = full_dataset.input_dim
    cfg.setdefault("model", {})
    cfg["model"]["input_dim"] = int(real_input_dim)
    logger.info(
        f"Detected input_dim={real_input_dim}, "
        f"set cfg['model']['input_dim'] accordingly."
    )
    # ==== 自动设置 input_dim 结束 ====

    # 构建模型
    model = FusionModel(cfg).to(device)

    # 构建损失函数
    loss_cfg = cfg.get("loss", {})
    criterion = PhysicsLoss(
        lambda_nonneg=loss_cfg.get("lambda_nonneg", 0.10),
        lambda_smooth=loss_cfg.get("lambda_smooth", 0.05),
        lambda_extreme=loss_cfg.get("lambda_extreme", 0.0),
        lambda_relative=loss_cfg.get("lambda_relative", 0.0),
        extreme_alpha=loss_cfg.get("extreme_alpha", 1.0),
    )

    # 优化器
    lr = float(cfg["training"]["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练
    train_model(model, dataloaders, criterion, optimizer, cfg, device, logger)


if __name__ == "__main__":
    main()