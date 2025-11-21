# hparam_search.py
import os
import argparse
import yaml
import torch
import pandas as pd
from copy import deepcopy
from datetime import datetime
from itertools import product

from models.fusion_model import FusionModel
from utils.dataset import get_dataloaders
from utils.physics_loss import PhysicsLoss
from utils.trainer import train_model
from utils.logger import get_file_logger

# ===========================
# 超参数搜索空间（按需修改）
# ===========================
SEARCH_SPACE = {
    "gas": [
        {"hidden_layers": [64, 32], "dropout": 0.05},
        {"hidden_layers": [128, 64, 32], "dropout": 0.10},
    ],
    "liquid": [
        {"hidden_layers": [128, 64, 32], "dropout": 0.10},
        {"hidden_layers": [256, 128, 64], "dropout": 0.15},
    ],
    "critical": [
        {"hidden_layers": [256, 128, 64], "dropout": 0.15},
        {"hidden_layers": [512, 256, 128], "dropout": 0.20},
    ],
    "extra": [
        {"hidden_layers": [128, 64, 32], "dropout": 0.10},
        {"hidden_layers": [256, 128, 64], "dropout": 0.15},
    ],
    "gate": [
        {"hidden_layers": [64, 32], "dropout": 0.05},
        {"hidden_layers": [128, 64], "dropout": 0.10},
    ],
    "training": {
        "learning_rate": [1e-3, 5e-4],
        "pretrain_epochs": [0, 20, 50],
        "batch_size": [64],
    },
}


def build_experiment_cfgs(base_cfg):
    gas_opts = SEARCH_SPACE["gas"]
    liq_opts = SEARCH_SPACE["liquid"]
    crit_opts = SEARCH_SPACE["critical"]
    extra_opts = SEARCH_SPACE.get("extra", [])
    gate_opts = SEARCH_SPACE["gate"]
    lr_list = SEARCH_SPACE["training"]["learning_rate"]
    pretrain_list = SEARCH_SPACE["training"]["pretrain_epochs"]
    bs_list = SEARCH_SPACE["training"].get("batch_size", [base_cfg["training"]["batch_size"]])

    if not extra_opts or "extra" not in base_cfg.get("experts", {}):
        extra_opts = [None]

    all_cfgs = []
    for idx, (g_cfg, l_cfg, c_cfg, e_cfg, gate_cfg, lr, pre, bs) in enumerate(
        product(gas_opts, liq_opts, crit_opts, extra_opts, gate_opts, lr_list, pretrain_list, bs_list),
        start=1,
    ):
        cfg = deepcopy(base_cfg)

        cfg["experts"]["gas"]["hidden_layers"] = g_cfg["hidden_layers"]
        cfg["experts"]["gas"]["dropout"] = g_cfg["dropout"]

        cfg["experts"]["liquid"]["hidden_layers"] = l_cfg["hidden_layers"]
        cfg["experts"]["liquid"]["dropout"] = l_cfg["dropout"]

        cfg["experts"]["critical"]["hidden_layers"] = c_cfg["hidden_layers"]
        cfg["experts"]["critical"]["dropout"] = c_cfg["dropout"]

        if e_cfg is not None and "extra" in cfg["experts"]:
            cfg["experts"]["extra"]["hidden_layers"] = e_cfg["hidden_layers"]
            cfg["experts"]["extra"]["dropout"] = e_cfg["dropout"]

        cfg["gate"]["hidden_layers"] = gate_cfg["hidden_layers"]
        cfg["gate"]["dropout"] = gate_cfg["dropout"]

        cfg.setdefault("training", {})
        cfg["training"]["learning_rate"] = float(lr)
        cfg["training"]["batch_size"] = int(bs)
        cfg["training"]["pretrain_epochs"] = int(pre)

        tag = (
            f"g{idx}_"
            f"G{g_cfg['hidden_layers']}_L{l_cfg['hidden_layers']}_"
            f"C{c_cfg['hidden_layers']}_"
            f"E{e_cfg['hidden_layers'] if e_cfg is not None else 'NA'}_"
            f"Gate{gate_cfg['hidden_layers']}_"
            f"lr{lr:g}_pre{pre}_bs{bs}"
        ).replace(" ", "")
        all_cfgs.append((tag, cfg))

    return all_cfgs


def run_single_experiment(cfg, tag, root_dir, base_data_path, log):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(root_dir, f"{tag}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    cfg["paths"]["save_dir"] = save_dir
    cfg["paths"]["scaler"] = os.path.join(save_dir, "scaler.pkl")
    cfg["paths"]["data"] = base_data_path

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

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    log_file = os.path.join(save_dir, "logs", "training.log")
    logger = get_file_logger(log_file, name=f"train_{tag}")
    logger.info(f"[{tag}] Device: {device}")

    train_model(model, dataloaders, criterion, optimizer, cfg, device, logger)

    metrics_path = os.path.join(save_dir, "plots", "training_metrics.csv")
    if os.path.exists(metrics_path):
        df_m = pd.read_csv(metrics_path)
        best_val = float(df_m["ValLoss"].min())
    else:
        best_val = float("nan")

    rec = {
        "Tag": tag,
        "SaveDir": save_dir,
        "ValLoss": best_val,
        "Gas_Layers": str(cfg["experts"]["gas"]["hidden_layers"]),
        "Gas_Dropout": cfg["experts"]["gas"]["dropout"],
        "Liq_Layers": str(cfg["experts"]["liquid"]["hidden_layers"]),
        "Liq_Dropout": cfg["experts"]["liquid"]["dropout"],
        "Crit_Layers": str(cfg["experts"]["critical"]["hidden_layers"]),
        "Crit_Dropout": cfg["experts"]["critical"]["dropout"],
        "Gate_Layers": str(cfg["gate"]["hidden_layers"]),
        "Gate_Dropout": cfg["gate"]["dropout"],
        "LearningRate": cfg["training"]["learning_rate"],
        "BatchSize": cfg["training"]["batch_size"],
        "PretrainEpochs": cfg["training"].get("pretrain_epochs", 0),
    }

    if "extra" in cfg["experts"]:
        rec["Extra_Layers"] = str(cfg["experts"]["extra"]["hidden_layers"])
        rec["Extra_Dropout"] = cfg["experts"]["extra"]["dropout"]
    else:
        rec["Extra_Layers"] = ""
        rec["Extra_Dropout"] = 0.0

    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="基础配置文件路径",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="results/hparam_search",
        help="超参搜索结果根目录",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="可选：覆盖 config 中 data 路径",
    )
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    log = get_file_logger(os.path.join(args.root, "hparam_search.log"), name="hparam_search")

    if not os.path.exists(args.config):
        log.info(f"Config file not found: {args.config}")
        return

    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    if args.data is not None:
        base_cfg["paths"]["data"] = args.data

    base_data_path = base_cfg["paths"]["data"]

    exp_cfgs = build_experiment_cfgs(base_cfg)
    log.info(f"Total experiments to run: {len(exp_cfgs)}")

    results = []
    for i, (tag, cfg) in enumerate(exp_cfgs, start=1):
        log.info(f"=== [{i}/{len(exp_cfgs)}] Running {tag} ===")
        rec = run_single_experiment(cfg, tag, args.root, base_data_path, log)
        results.append(rec)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_all = pd.DataFrame(results)
    df_all = df_all.sort_values(by="ValLoss", ascending=True)
    out_path = os.path.join(args.root, "summary_table.csv")
    df_all.to_csv(out_path, index=False)
    log.info(f"Search complete. Summary written to {out_path}")


if __name__ == "__main__":
    main()
