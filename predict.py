# predict.py
import os
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

from models.fusion_model import FusionModel
from utils.dataset import ZDataset
from utils.logger import get_file_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径（需要和训练时一致）"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = cfg["paths"]["data"]
    scaler_path = cfg["paths"]["scaler"]
    target_col = cfg.get("target_col", "Z (-)")
    expert_col = cfg.get("expert_col", "no")
    subset_cfg = cfg.get("subset", None)

    save_dir = cfg["paths"]["save_dir"]
    eval_dir = os.path.join(save_dir, "eval")
    region_dir = os.path.join(save_dir, "region_eval")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(region_dir, exist_ok=True)

    logger = get_file_logger(os.path.join(eval_dir, "eval.log"), name="eval")
    logger.info(f"Device: {device}")

    # 用 train=False 加载数据，复用训练时的 scaler
    dataset = ZDataset(
        csv_path=data_path,
        scaler_path=scaler_path,
        train=False,
        target_col=target_col,
        expert_col=expert_col,
        subset_cfg=subset_cfg,
    )

    # 让模型结构与数据维度一致
    cfg.setdefault("model", {})
    cfg["model"]["input_dim"] = dataset.input_dim
    logger.info(
        f"Dataset input_dim = {dataset.input_dim}, "
        f"set cfg['model']['input_dim'] accordingly."
    )

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    g = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=g
    )

    batch_size = cfg["training"]["batch_size"]
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    full_ids = dataset.expert_ids
    test_indices = np.array(test_set.indices, dtype=int)
    test_expert_ids = full_ids[test_indices]

    model = FusionModel(cfg).to(device)
    ckpt_path = os.path.join(save_dir, "checkpoints", "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, y, _ = batch
            elif len(batch) == 2:
                x, y = batch
            else:
                raise RuntimeError("Unexpected batch format in eval.")
            x = x.to(device)
            y = y.to(device)
            preds, _, _ = model(x)

            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0).reshape(-1)
    y_pred = np.concatenate(y_pred_list, axis=0).reshape(-1)

    if len(test_expert_ids) != len(y_true):
        logger.warning(
            f"Length mismatch between test_expert_ids ({len(test_expert_ids)}) and y_true ({len(y_true)}). "
            "Will truncate to the smaller length."
        )
        n_min = min(len(test_expert_ids), len(y_true))
        test_expert_ids = test_expert_ids[:n_min]
        y_true = y_true[:n_min]
        y_pred = y_pred[:n_min]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logger.info(f"Test MAE={mae:.6f} MSE={mse:.6f} R2={r2:.6f}")

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "R2": float(r2),
        "N_test": int(len(y_true)),
    }
    with open(os.path.join(eval_dir, "metrics_summary.yaml"), "w") as f:
        yaml.dump(metrics, f)

    df_pred = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "expert_id": test_expert_ids.astype(int),
    })
    pred_csv_path = os.path.join(eval_dir, "test_predictions.csv")
    df_pred.to_csv(pred_csv_path, index=False)

    df_pred["expert_id_str"] = df_pred["expert_id"].astype(str)
    color_map = {
        "1": "#E699A7",
        "2": "#FEDD9E",
        "3": "#A6D9C0",
        "4": "#71A7D2",
    }

    fig = px.scatter(
        df_pred,
        x="y_true",
        y="y_pred",
        color="expert_id_str",
        color_discrete_map=color_map,
        title="True vs Predicted on Test Set (Colored by Expert ID)",
        labels={
            "y_true": "True Z",
            "y_pred": "Predicted Z",
            "expert_id_str": "Expert ID (no)",
        },
        width=750,
        height=700,
    )

    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(dash="dash"),
    )
    fig.update_layout(xaxis_range=[min_val, max_val], yaxis_range=[min_val, max_val])

    scatter_path = os.path.join(eval_dir, "true_vs_pred_scatter.html")
    fig.write_html(scatter_path)

    region_scatter_path = os.path.join(region_dir, "true_vs_pred_by_region.html")
    fig.write_html(region_scatter_path)

    region_metrics = []
    unique_ids = sorted(df_pred["expert_id"].unique())
    for eid in unique_ids:
        df_g = df_pred[df_pred["expert_id"] == eid]
        y_true_g = df_g["y_true"].values
        y_pred_g = df_g["y_pred"].values

        mae_g = mean_absolute_error(y_true_g, y_pred_g)
        mse_g = mean_squared_error(y_true_g, y_pred_g)
        r2_g = r2_score(y_true_g, y_pred_g)

        region_metrics.append({
            "expert_id": int(eid),
            "N": int(len(df_g)),
            "MAE": float(mae_g),
            "MSE": float(mse_g),
            "R2": float(r2_g),
        })

    region_metrics_df = pd.DataFrame(region_metrics)
    region_metrics_csv = os.path.join(region_dir, "region_metrics.csv")
    region_metrics_df.to_csv(region_metrics_csv, index=False)

    region_metrics_yaml = os.path.join(region_dir, "region_metrics.yaml")
    with open(region_metrics_yaml, "w") as f:
        yaml.dump({"regions": region_metrics}, f)

    logger.info(f"Region-wise metrics saved to {region_dir}")

    summary_path = os.path.join(save_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    summary.setdefault("TestMetrics", {})
    summary["TestMetrics"].update({
        "MAE": float(mae),
        "MSE": float(mse),
        "R2": float(r2),
        "N_test": int(len(y_true)),
    })

    summary.setdefault("RegionMetrics", {})
    for rm in region_metrics:
        key = str(rm["expert_id"])
        summary["RegionMetrics"][key] = {
            "N": rm["N"],
            "MAE": rm["MAE"],
            "MSE": rm["MSE"],
            "R2": rm["R2"],
        }

    summary.setdefault("EvalArtifacts", {})
    summary["EvalArtifacts"].update({
        "pred_csv": os.path.abspath(pred_csv_path),
        "scatter_html": os.path.abspath(scatter_path),
    })
    summary.setdefault("RegionArtifacts", {})
    summary["RegionArtifacts"].update({
        "region_scatter_html": os.path.abspath(region_scatter_path),
        "region_metrics_csv": os.path.abspath(region_metrics_csv),
        "region_metrics_yaml": os.path.abspath(region_metrics_yaml),
    })

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evaluation artifacts saved to {eval_dir} and {region_dir}")
    logger.info(f"Summary updated at {summary_path}")


if __name__ == "__main__":
    main()
