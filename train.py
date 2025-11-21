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

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.data is not None:
        cfg["paths"]["data"] = args.data

    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    cfg["paths"]["scaler"] = cfg["paths"].get("scaler", os.path.join(save_dir, "scaler.pkl"))

    # 记录本次 config
    with open(os.path.join(save_dir, "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloaders = get_dataloaders(cfg)

    model = FusionModel(cfg).to(device)
    loss_cfg = cfg.get("loss", {})
    criterion = PhysicsLoss(
        lambda_nonneg=loss_cfg.get("lambda_nonneg", 0.10),
        lambda_smooth=loss_cfg.get("lambda_smooth", 0.05),
    )

    # 这里 lr 显式转换为 float，避免 "1e-3" 字符串类型报错
    lr = float(cfg["training"]["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_file = os.path.join(save_dir, "logs", "training.log")
    logger = get_file_logger(log_file, name="train")
    logger.info(f"Device: {device}")

    train_model(model, dataloaders, criterion, optimizer, cfg, device, logger)


if __name__ == "__main__":
    main()
