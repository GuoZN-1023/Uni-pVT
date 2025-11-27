# optimize.py
"""
使用贝叶斯优化（Optuna）对 Uni-pVT 的超参数进行多目标优化（帕累托前沿）。

做的事情：
    1. 从基础配置 config/config.yaml 复制一份；
    2. 为每个 trial 采样一组超参数（激活函数 / batch_size / lr / lambda_*）；
    3. 写入 trial 专属 config + save_dir；
    4. 调用 train.py / predict.py 完成训练 + 评估；
    5. 读取 eval/metrics_summary.yaml 提取 R2 / MAE / MSE；
    6. 将 (1-R2, MAE, MSE) 作为三目标最小化，Optuna 搜索帕累托前沿；
    7. 优化结果保存在 optuna_results/ 下，包括 Pareto trial 的参数与指标。

用法示例：
    python optimize.py --config config/config.yaml --n-trials 30

注意事项：
    - 需要安装 optuna： pip install optuna
    - 训练和预测日志仍由 train.py / predict.py 写入各自的 save_dir/logs 下；
      optimize.py 本身只在终端打印简要进度。
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime

import yaml
import optuna


# 只在这三种激活之间选
ACTIVATIONS = ["relu", "gelu", "tanh"]


def run_subprocess(cmd, stdout_path=None, stderr_path=None, cwd=None):
    """
    小工具：运行子进程。
    如果 stdout_path/stderr_path 提供，则将输出重定向到文件，避免刷屏。
    """
    stdout = None
    stderr = None
    if stdout_path is not None:
        os.makedirs(os.path.dirname(stdout_path), exist_ok=True)
        stdout = open(stdout_path, "a", encoding="utf-8")
    if stderr_path is not None:
        os.makedirs(os.path.dirname(stderr_path), exist_ok=True)
        stderr = open(stderr_path, "a", encoding="utf-8")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=stdout if stdout is not None else None,
            stderr=stderr if stderr is not None else None,
            check=True,
        )
    finally:
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()
    return result.returncode


def load_metrics(metrics_path: str):
    """
    从 metrics_summary.yaml 中读取 R2 / MAE / MSE。
    根据你之前 predict.py 的写法，可能有几种命名，我们做一点兜底。
    """
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 可能有多层结构，比如 {"overall": {"r2": ...}}
    if isinstance(data, dict):
        # 优先找 overall
        if "overall" in data and isinstance(data["overall"], dict):
            d = data["overall"]
        else:
            d = data
    else:
        raise ValueError(f"Unexpected metrics format in {metrics_path}")

    def get_with_aliases(dct, aliases, default=None):
        for k in aliases:
            if k in dct:
                return float(dct[k])
        return default

    r2 = get_with_aliases(d, ["r2", "R2", "R^2"])
    mae = get_with_aliases(d, ["mae", "MAE"])
    mse = get_with_aliases(d, ["mse", "MSE"])
    if r2 is None or mae is None or mse is None:
        raise ValueError(f"Missing metrics in {metrics_path}, got keys: {list(d.keys())}")

    return r2, mae, mse


def make_trial_config(base_cfg, trial: optuna.trial.Trial, trial_dir: str):
    """
    基于基础 config + trial 超参数，生成本次试验的 config dict。
    这里只调：
      - training.batch_size
      - training.learning_rate
      - loss.lambda_nonneg / lambda_smooth
      - experts/gate 的激活函数（统一为一个激活，不按层细分，先控制复杂度）
    """
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # 深拷贝

    # 搜索空间
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    lr = trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True)

    lambda_nonneg = trial.suggest_float("lambda_nonneg", 1e-6, 1e-1, log=True)
    lambda_smooth = trial.suggest_float("lambda_smooth", 1e-6, 1e-1, log=True)

    act_gas = trial.suggest_categorical("act_gas", ACTIVATIONS)
    act_liq = trial.suggest_categorical("act_liquid", ACTIVATIONS)
    act_crit = trial.suggest_categorical("act_critical", ACTIVATIONS)
    act_extra = trial.suggest_categorical("act_extra", ACTIVATIONS)
    act_gate = trial.suggest_categorical("act_gate", ACTIVATIONS)

    # 写回 config
    cfg.setdefault("training", {})
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["learning_rate"] = lr

    cfg.setdefault("loss", {})
    cfg["loss"]["lambda_nonneg"] = lambda_nonneg
    cfg["loss"]["lambda_smooth"] = lambda_smooth

    cfg.setdefault("experts", {})
    for key, act in [
        ("gas", act_gas),
        ("liquid", act_liq),
        ("critical", act_crit),
        ("extra", act_extra),
    ]:
        cfg["experts"].setdefault(key, {})
        cfg["experts"][key]["activation"] = act

    cfg.setdefault("gate", {})
    cfg["gate"]["activation"] = act_gate

    # 每个 trial 独立的 save_dir / scaler
    cfg.setdefault("paths", {})
    cfg["paths"]["save_dir"] = trial_dir
    cfg["paths"]["scaler"] = os.path.join(trial_dir, "scaler.pkl")

    return cfg


def create_objective(base_config_path: str, optuna_root: str, python_exec: str):
    """
    返回给 Optuna 用的 objective 函数。
    """

    # 预先加载基础配置
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs(optuna_root, exist_ok=True)

    def objective(trial: optuna.trial.Trial):
        trial_id = f"trial_{trial.number:04d}"
        trial_dir = os.path.join(optuna_root, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        logs_dir = os.path.join(trial_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # 为本次 trial 构建 config
        cfg = make_trial_config(base_cfg, trial, trial_dir)
        trial_config_path = os.path.join(trial_dir, "config_trial.yaml")
        with open(trial_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

        print(f"[Optuna] Start {trial_id} -> {trial_config_path}")

        # 调用 train.py
        train_stdout = os.path.join(logs_dir, "train.stdout.log")
        train_stderr = os.path.join(logs_dir, "train.stderr.log")
        train_cmd = [python_exec, "train.py", "--config", trial_config_path]
        try:
            run_subprocess(train_cmd, stdout_path=train_stdout, stderr_path=train_stderr)
        except subprocess.CalledProcessError as e:
            print(f"[Optuna] {trial_id} train failed: {e}")
            # 你也可以改成 TrialPruned 或 Fail，这里直接抛异常让 Optuna 记为失败
            raise

        # 调用 predict.py
        pred_stdout = os.path.join(logs_dir, "predict.stdout.log")
        pred_stderr = os.path.join(logs_dir, "predict.stderr.log")
        pred_cmd = [python_exec, "predict.py", "--config", trial_config_path]
        try:
            run_subprocess(pred_cmd, stdout_path=pred_stdout, stderr_path=pred_stderr)
        except subprocess.CalledProcessError as e:
            print(f"[Optuna] {trial_id} predict failed: {e}")
            raise

        # 读取 metrics
        metrics_path = os.path.join(trial_dir, "eval", "metrics_summary.yaml")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"metrics_summary.yaml not found for {trial_id} at {metrics_path}")

        r2, mae, mse = load_metrics(metrics_path)
        print(f"[Optuna] {trial_id} metrics: R2={r2:.6f}, MAE={mae:.6e}, MSE={mse:.6e}")

        # 记录到本 trial 的 summary.json（方便之后分析）
        summary_json = os.path.join(trial_dir, "optuna_summary.json")
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "trial_number": trial.number,
                    "params": trial.params,
                    "metrics": {"r2": r2, "mae": mae, "mse": mse},
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # 多目标：三个都是要“越小越好”
        #   1 - R2：R2 越大越好，所以 1-R2 越小越好
        #   MAE：越小越好
        #   MSE：越小越好
        return 1.0 - r2, mae, mse

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="基础配置文件路径（会作为模板拷贝）",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Optuna 试验次数",
    )
    parser.add_argument(
        "--optuna-dir",
        type=str,
        default="optuna_results",
        help="所有 trial 结果保存的根目录",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study 名称（可选，用于持久化到数据库时区分）",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage，例如 'sqlite:///optuna.db'，不需要持久化可留空",
    )

    args = parser.parse_args()

    python_exec = sys.executable
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    optuna_root = os.path.join(args.optuna_dir, timestamp)
    os.makedirs(optuna_root, exist_ok=True)

    print(f"[Optuna] Base config:  {os.path.abspath(args.config)}")
    print(f"[Optuna] Results root: {os.path.abspath(optuna_root)}")
    print(f"[Optuna] Python exec:  {python_exec}")

    # 多目标 Study
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],  # (1-R2, MAE, MSE)
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage and args.study_name),
        sampler=optuna.samplers.TPESampler(multivariate=True),
    )

    objective = create_objective(args.config, optuna_root, python_exec)

    study.optimize(objective, n_trials=args.n_trials)

    print("\n[Optuna] Optimization finished.")
    print(f"  Number of finished trials: {len(study.trials)}")

    # 对于多目标，best_trials 就是目前的非支配解集合（帕累托前沿）
    pareto_trials = study.best_trials
    print(f"  Number of Pareto-optimal trials: {len(pareto_trials)}")

    # 写一个汇总 JSON
    pareto_summary = []
    for t in pareto_trials:
        v1, v2, v3 = t.values  # (1-R2, MAE, MSE)
        pareto_summary.append(
            {
                "trial_number": t.number,
                "params": t.params,
                "objectives": {
                    "1-R2": v1,
                    "MAE": v2,
                    "MSE": v3,
                },
            }
        )

    pareto_json_path = os.path.join(optuna_root, "pareto_front.json")
    with open(pareto_json_path, "w", encoding="utf-8") as f:
        json.dump(pareto_summary, f, indent=2, ensure_ascii=False)

    print(f"[Optuna] Pareto front saved to: {os.path.abspath(pareto_json_path)}")
    print("  You can open each trial_xxxx/optuna_summary.json for full details.")


if __name__ == "__main__":
    main()