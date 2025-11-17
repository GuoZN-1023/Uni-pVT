import os
import yaml
import pandas as pd
import ast
from datetime import datetime
from utils.logger import get_file_logger

def update_best_config(summary_path="results/hparam_search/summary_table.csv", config_path="configs/config.yaml"):
    logger = get_file_logger("results/hparam_search/hparam_update.log", name="hparam_update")

    if not os.path.exists(summary_path):
        logger.info(f"Not found: {summary_path}")
        return
    if not os.path.exists(config_path):
        logger.info(f"Not found: {config_path}")
        return

    df = pd.read_csv(summary_path)
    if "ValLoss" not in df.columns:
        logger.info("ValLoss column missing.")
        return

    best_row = df.loc[df["ValLoss"].idxmin()]
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    backup = config_path.replace(".yaml", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    with open(backup, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    cfg["experts"]["gas"]["hidden_layers"] = ast.literal_eval(str(best_row["Gas_Layers"]))
    cfg["experts"]["liquid"]["hidden_layers"] = ast.literal_eval(str(best_row["Liq_Layers"]))
    cfg["experts"]["critical"]["hidden_layers"] = ast.literal_eval(str(best_row["Crit_Layers"]))

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    logger.info(f"Best structure updated to {config_path}, backup at {backup}. "
                f"Best ValLoss={best_row['ValLoss']:.6f}")

if __name__ == "__main__":
    update_best_config()