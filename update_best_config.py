# update_best_config.py
import os
import yaml
import pandas as pd
import ast
from datetime import datetime
from utils.logger import get_file_logger


def update_best_config(
    summary_path: str = "results/hparam_search/summary_table.csv",
    config_path: str = "config/config.yaml",
):
    """
    从 hparam_search 的 summary_table.csv 中选出 ValLoss 最小的一行，
    把对应的结构和超参数写回主 config.yaml。
    """
    logger = get_file_logger(
        os.path.join(os.path.dirname(summary_path), "hparam_update.log"),
        name="hparam_update"
    )

    if not os.path.exists(summary_path):
        logger.info(f"Not found: {summary_path}")
        return
    if not os.path.exists(config_path):
        logger.info(f"Not found: {config_path}")
        return

    df = pd.read_csv(summary_path)
    if df.empty:
        logger.info("summary_table.csv is empty.")
        return

    best_row = df.sort_values(by="ValLoss", ascending=True).iloc[0]

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = config_path.replace(".yaml", f".backup_{ts}.yaml")
    with open(backup, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    def parse_layers(v):
        try:
            return list(ast.literal_eval(str(v)))
        except Exception:
            return []

    cfg["experts"]["gas"]["hidden_layers"] = parse_layers(best_row.get("Gas_Layers", "[]"))
    cfg["experts"]["gas"]["dropout"] = float(best_row.get("Gas_Dropout", cfg["experts"]["gas"].get("dropout", 0.0)))

    cfg["experts"]["liquid"]["hidden_layers"] = parse_layers(best_row.get("Liq_Layers", "[]"))
    cfg["experts"]["liquid"]["dropout"] = float(best_row.get("Liq_Dropout", cfg["experts"]["liquid"].get("dropout", 0.0)))

    cfg["experts"]["critical"]["hidden_layers"] = parse_layers(best_row.get("Crit_Layers", "[]"))
    cfg["experts"]["critical"]["dropout"] = float(best_row.get("Crit_Dropout", cfg["experts"]["critical"].get("dropout", 0.0)))

    if "Extra_Layers" in best_row and "extra" in cfg.get("experts", {}):
        cfg["experts"]["extra"]["hidden_layers"] = parse_layers(best_row.get("Extra_Layers", "[]"))
        cfg["experts"]["extra"]["dropout"] = float(best_row.get("Extra_Dropout", cfg["experts"]["extra"].get("dropout", 0.0)))

    cfg["gate"]["hidden_layers"] = parse_layers(best_row.get("Gate_Layers", "[]"))
    cfg["gate"]["dropout"] = float(best_row.get("Gate_Dropout", cfg["gate"].get("dropout", 0.0)))

    cfg.setdefault("training", {})
    if "LearningRate" in best_row:
        cfg["training"]["learning_rate"] = float(best_row["LearningRate"])
    if "BatchSize" in best_row:
        cfg["training"]["batch_size"] = int(best_row["BatchSize"])
    if "PretrainEpochs" in best_row:
        cfg["training"]["pretrain_epochs"] = int(best_row["PretrainEpochs"])

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    logger.info(
        f"Best config updated to {config_path}, backup at {backup}. "
        f"Best ValLoss={best_row['ValLoss']:.6f}"
    )


if __name__ == "__main__":
    update_best_config()
