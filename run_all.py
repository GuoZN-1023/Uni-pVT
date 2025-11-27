# run_all.py
import os
import sys
import json
import shutil
import argparse
import traceback
from datetime import datetime
import yaml
import subprocess


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _append(path: str, msg: str):
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg if msg.endswith("\n") else msg + "\n")


def run_cmd(cmd, stdout_path: str, stderr_path: str, runall_log: str, cwd=None, allow_fail=False):
    _append(runall_log, f"\n[run_all] Running: {' '.join(cmd)}")
    _append(runall_log, f"[run_all]  stdout -> {stdout_path}")
    _append(runall_log, f"[run_all]  stderr -> {stderr_path}")

    _ensure_dir(os.path.dirname(stdout_path))
    _ensure_dir(os.path.dirname(stderr_path))

    with open(stdout_path, "a", encoding="utf-8") as out_f, open(stderr_path, "a", encoding="utf-8") as err_f:
        out_f.write(f"\n========== CMD START: {' '.join(cmd)} ==========\n")
        err_f.write(f"\n========== CMD START: {' '.join(cmd)} ==========\n")
        out_f.flush(); err_f.flush()

        result = subprocess.run(cmd, cwd=cwd, stdout=out_f, stderr=err_f)

        out_f.write(f"========== CMD END (returncode={result.returncode}) ==========\n")
        err_f.write(f"========== CMD END (returncode={result.returncode}) ==========\n")
        out_f.flush(); err_f.flush()

    _append(runall_log, f"[run_all] Return code: {result.returncode}")

    if result.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--shap_strict", action="store_true", help="SHAP失败则整次任务失败（默认：SHAP失败只记录日志）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_results_dir = base_cfg.get("paths", {}).get("root_results", "results")
    exp_dir = os.path.join(root_results_dir, timestamp)
    logs_dir = os.path.join(exp_dir, "logs")
    _ensure_dir(exp_dir); _ensure_dir(logs_dir); _ensure_dir(os.path.join(exp_dir, "checkpoints"))

    runall_log = os.path.join(logs_dir, "run_all.log")
    _append(runall_log, "========== RUN_ALL START ==========")
    _append(runall_log, f"timestamp: {timestamp}")
    _append(runall_log, f"exp_dir:    {os.path.abspath(exp_dir)}")
    _append(runall_log, f"base_config:{os.path.abspath(args.config)}")

    cfg = dict(base_cfg)
    cfg.setdefault("paths", {})
    cfg["paths"]["save_dir"] = exp_dir
    cfg["paths"]["scaler"] = os.path.join(exp_dir, "scaler.pkl")

    exp_config_path = os.path.join(exp_dir, "config_used.yaml")
    with open(exp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    _ensure_dir("config")
    shutil.copy(exp_config_path, os.path.join("config", "current.yaml"))

    python_exec = sys.executable
    train_out = os.path.join(logs_dir, "train.stdout.log")
    train_err = os.path.join(logs_dir, "train.stderr.log")
    pred_out = os.path.join(logs_dir, "predict.stdout.log")
    pred_err = os.path.join(logs_dir, "predict.stderr.log")
    shap_out = os.path.join(logs_dir, "shap.stdout.log")
    shap_err = os.path.join(logs_dir, "shap.stderr.log")

    _append(runall_log, "\n===== STAGE: TRAIN =====")
    run_cmd([python_exec, "train.py", "--config", exp_config_path],
            stdout_path=train_out, stderr_path=train_err, runall_log=runall_log)

    _append(runall_log, "\n===== STAGE: PREDICT =====")
    run_cmd([python_exec, "predict.py", "--config", exp_config_path],
            stdout_path=pred_out, stderr_path=pred_err, runall_log=runall_log)

    shap_cfg = cfg.get("shap", {})
    shap_enabled = bool(shap_cfg.get("enabled", False))
    _append(runall_log, f"\n===== STAGE: SHAP (enabled={shap_enabled}, strict={args.shap_strict}) =====")

    shap_rc = None
    if shap_enabled:
        shap_rc = run_cmd([python_exec, "shap_analysis.py", "--config", exp_config_path],
                          stdout_path=shap_out, stderr_path=shap_err, runall_log=runall_log,
                          allow_fail=(not args.shap_strict))
        if shap_rc != 0:
            _append(runall_log, f"[run_all] SHAP failed (rc={shap_rc}). See:\n  {shap_err}\n  {os.path.join(exp_dir, 'shap', 'crash.log')}")
    else:
        _append(runall_log, "[run_all] SHAP disabled. Set shap.enabled: true in config to enable.")

    summary_path = os.path.join(exp_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    summary.setdefault("RunAll", {})
    summary["RunAll"].update({
        "timestamp": timestamp,
        "exp_dir": os.path.abspath(exp_dir),
        "config_used": os.path.abspath(exp_config_path),
        "run_all_log": os.path.abspath(runall_log),
        "shap_enabled": shap_enabled,
        "shap_returncode": shap_rc,
        "logs": {
            "train_stdout": os.path.abspath(train_out),
            "train_stderr": os.path.abspath(train_err),
            "predict_stdout": os.path.abspath(pred_out),
            "predict_stderr": os.path.abspath(pred_err),
            "shap_stdout": os.path.abspath(shap_out),
            "shap_stderr": os.path.abspath(shap_err),
        },
    })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _append(runall_log, "\n✅ RUN_ALL FINISHED")
    _append(runall_log, f"summary.json: {os.path.abspath(summary_path)}")
    _append(runall_log, "========== RUN_ALL END ==========")


if __name__ == "__main__":
    # 不往终端打印 traceback（写入 logs/run_all.log）
    try:
        main()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        # 尽量写到最新一次 results/*/logs/run_all.log
        try:
            root = "results"
            candidate = "run_all_error.log"
            if os.path.isdir(root):
                subdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
                if subdirs:
                    candidate = os.path.join(root, subdirs[-1], "logs", "run_all.log")
            _append(candidate, "\n❌ RUN_ALL FAILED")
            _append(candidate, tb)
        except Exception:
            pass
        sys.exit(1)