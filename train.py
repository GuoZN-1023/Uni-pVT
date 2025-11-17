import argparse
import yaml
import torch
from models.fusion_model import FusionModel
from utils.dataset import get_dataloaders
from utils.physics_loss import PhysicsLoss
from utils.trainer import train_model
from utils.logger import get_file_logger
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders = get_dataloaders(cfg)
    model = FusionModel(cfg).to(device)
    criterion = PhysicsLoss(cfg["loss"]["lambda_nonneg"], cfg["loss"]["lambda_smooth"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    log_file = os.path.join(cfg["paths"]["save_dir"], "logs", "training.log")
    logger = get_file_logger(log_file, name="train")
    logger.info(f"Device: {device}")
    logger.info(f"Config loaded from: {args.config}")

    train_model(model, dataloaders, criterion, optimizer, cfg, device, logger)

if __name__ == "__main__":
    main()