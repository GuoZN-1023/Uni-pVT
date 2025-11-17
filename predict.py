import argparse
import yaml
import torch
import numpy as np
from models.fusion_model import FusionModel
from utils.dataset import ZDataset
from utils.visualize import plot_predictions
from utils.phase_visualizer import visualize_phase_distribution
from utils.logger import get_file_logger
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(cfg).to(device)

    ckpt = os.path.join(cfg["paths"]["save_dir"], "checkpoints", "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    log_file = os.path.join(cfg["paths"]["save_dir"], "logs", "predict.log")
    logger = get_file_logger(log_file, name="predict")

    dataset = ZDataset(cfg["paths"]["data"], cfg["paths"]["scaler"], train=False)
    X = dataset.X.to(device)
    y_true = dataset.y.cpu().numpy().flatten()

    with torch.no_grad():
        y_pred, _, _ = model(X)
    y_pred = y_pred.cpu().numpy().flatten()

    plot_predictions(y_true, y_pred, f"{cfg['paths']['save_dir']}/results/prediction_comparison.html")
    visualize_phase_distribution(model, dataset, f"{cfg['paths']['save_dir']}/plots/phase_distribution.html", method="tsne", device=device)
    logger.info("Prediction complete. Outputs written to results/ & plots/.")

if __name__ == "__main__":
    main()