import subprocess
import datetime
import os
import sys
import shutil
import json
import traceback
import yaml
import torch
import pandas as pd

def write_log(message, log_file):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def run_command(command, log_file, section_name, cwd):
    write_log(f"\n===== 【{section_name}】开始 ===== {datetime.datetime.now()}\n", log_file)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=cwd)
        if result.stdout:
            write_log(result.stdout, log_file)
        if result.stderr:
            write_log("\n[stderr]\n" + result.stderr, log_file)
        write_log(f"\n===== 【{section_name}】结束 ===== {datetime.datetime.now()}\n", log_file)
    except Exception as e:
        write_log(f"\n[EXCEPTION]: {e}\n{traceback.format_exc()}\n", log_file)

def get_device_info():
    info = {
        "Python Version": sys.version.split()[0],
        "Torch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }
    return info

def main():
    # 根目录 results
    base_root = "results"
    os.makedirs(base_root, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(base_root, timestamp)
    for d in ["checkpoints", "logs", "plots", "results"]:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)

    log_file = os.path.join(base_dir, "logs", "run.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"========== 实验日志开始 [{datetime.datetime.now()}] ==========\n")

    # 复制并重写配置
    config_src = "configs/config.yaml"
    config_dst = os.path.join(base_dir, "config_copy.yaml")
    with open(config_src, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["save_dir"] = base_dir
    cfg["paths"]["scaler"] = os.path.join(base_dir, "scaler.pkl")
    with open(config_dst, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    write_log(f"已复制并注入配置到 {config_dst}", log_file)
    write_log(f"设备信息: {json.dumps(get_device_info(), ensure_ascii=False)}", log_file)

    # 训练
    run_command(f"{sys.executable} train.py --config {config_dst}", log_file, "模型训练阶段", cwd=os.getcwd())
    # 预测
    run_command(f"{sys.executable} predict.py --config {config_dst}", log_file, "预测与可视化阶段", cwd=os.getcwd())

    # 汇总 summary
    summary = {
        "Experiment_ID": timestamp,
        "Start_Time": timestamp,
        "Config_File": config_dst,
        "Environment": get_device_info(),
        "Artifacts": {
            "Log_File": log_file,
            "Checkpoints": os.path.join(base_dir, "checkpoints"),
            "Plots": os.path.join(base_dir, "plots"),
            "Results": os.path.join(base_dir, "results")
        }
    }

    metrics_csv = os.path.join(base_dir, "plots", "training_metrics.csv")
    if os.path.exists(metrics_csv):
        try:
            df = pd.read_csv(metrics_csv)
            best_row = df.loc[df["ValLoss"].idxmin()]
            summary["Best_Validation_Loss"] = float(best_row["ValLoss"])
            summary["Final_Gating_Weights"] = {
                "Gas": float(best_row["Gate_Gas"]),
                "Liquid": float(best_row["Gate_Liquid"]),
                "Critical": float(best_row["Gate_Critical"]),
            }
        except Exception as e:
            summary["Metrics_Extraction_Error"] = str(e)

    summary_path = os.path.join(base_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    write_log(f"\n✅ 实验完成，摘要已保存至 {summary_path}", log_file)
    write_log(f"📁 实验目录：{base_dir}", log_file)

if __name__ == "__main__":
    main()