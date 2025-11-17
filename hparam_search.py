import os
import yaml
import torch
import pandas as pd
from datetime import datetime
from models.fusion_model import FusionModel
from utils.dataset import get_dataloaders
from utils.physics_loss import PhysicsLoss
from utils.trainer import train_model
from utils.logger import get_file_logger

# 小搜索空间（低算力友好）
SEARCH_SPACE = {
    "gas": [
        {"hidden_layers": [64, 32], "dropout": 0.05},
        {"hidden_layers": [128, 64, 32], "dropout": 0.10},
    ],
    "liquid": [
        {"hidden_layers": [128, 64], "dropout": 0.10},
        {"hidden_layers": [256, 128, 64], "dropout": 0.15},
    ],
    "critical": [
        {"hidden_layers": [256, 128], "dropout": 0.15},
        {"hidden_layers": [512, 256, 128], "dropout": 0.20},
    ]
}

def run_experiment(cfg, tag, exp_dir, log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["paths"]["save_dir"] = exp_dir
    cfg["paths"]["scaler"] = os.path.join(exp_dir, "scaler.pkl")

    logger = get_file_logger(os.path.join(exp_dir, "logs", "training.log"), name=tag)

    dataloaders = get_dataloaders(cfg)
    model = FusionModel(cfg).to(device)
    criterion = PhysicsLoss(cfg["loss"]["lambda_nonneg"], cfg["loss"]["lambda_smooth"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    train_model(model, dataloaders, criterion, optimizer, cfg, device, logger)

    metrics_path = os.path.join(exp_dir, "plots", "training_metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        best_row = df.loc[df["ValLoss"].idxmin()]
        record = {
            "Tag": tag,
            "ValLoss": float(best_row["ValLoss"]),
            "Gas_Layers": cfg["experts"]["gas"]["hidden_layers"],
            "Liq_Layers": cfg["experts"]["liquid"]["hidden_layers"],
            "Crit_Layers": cfg["experts"]["critical"]["hidden_layers"],
            "Gate": cfg["gate"]["hidden_layers"],
            "Gas_W": float(best_row["Gate_Gas"]),
            "Liq_W": float(best_row["Gate_Liquid"]),
            "Crit_W": float(best_row["Gate_Critical"]),
            "Exp_Dir": exp_dir
        }
    else:
        record = {"Tag": tag, "ValLoss": None, "Exp_Dir": exp_dir}
    log.info(f"{tag} result: {record}")
    return record

def main():
    # 读取基础配置
    with open("configs/config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    # 降低训练负担
    base_cfg["training"]["epochs"] = 50
    base_cfg["training"]["early_stopping_patience"] = 10

    root = "results/hparam_search"
    os.makedirs(root, exist_ok=True)
    log = get_file_logger(os.path.join(root, "hparam_search.log"), name="hparam")

    results = []
    total_runs = len(SEARCH_SPACE["gas"]) * len(SEARCH_SPACE["liquid"]) * len(SEARCH_SPACE["critical"])
    run_id = 0

    for g in SEARCH_SPACE["gas"]:
        for l in SEARCH_SPACE["liquid"]:
            for c in SEARCH_SPACE["critical"]:
                run_id += 1
                tag = f"Exp_{run_id:02d}"
                exp_dir = os.path.join(root, f"{tag}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
                os.makedirs(exp_dir, exist_ok=True)

                cfg = yaml.safe_load(yaml.dump(base_cfg))  # 深拷贝
                cfg["experts"]["gas"]["hidden_layers"] = g["hidden_layers"]
                cfg["experts"]["gas"]["dropout"] = g["dropout"]
                cfg["experts"]["liquid"]["hidden_layers"] = l["hidden_layers"]
                cfg["experts"]["liquid"]["dropout"] = l["dropout"]
                cfg["experts"]["critical"]["hidden_layers"] = c["hidden_layers"]
                cfg["experts"]["critical"]["dropout"] = c["dropout"]

                log.info(f"Running {tag} with cfg: {cfg['experts']}")
                rec = run_experiment(cfg, tag, exp_dir, log)
                results.append(rec)

                # 释放显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    df_all = pd.DataFrame(results)
    df_all = df_all.sort_values(by="ValLoss", ascending=True)
    df_all.to_csv(os.path.join(root, "summary_table.csv"), index=False)
    log.info("Search complete. Summary written to summary_table.csv")

if __name__ == "__main__":
    main()