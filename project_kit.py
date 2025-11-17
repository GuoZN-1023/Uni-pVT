import argparse
import subprocess
import sys
import os

def run_script(script, args=None):
    cmd = [sys.executable, script] + (args or [])
    # 静默运行：丢弃输出（各脚本自行写日志）
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

def main():
    parser = argparse.ArgumentParser(description="Project1 统一科研命令入口（静默）")
    parser.add_argument("mode", choices=["train", "predict", "search", "update", "all"], help="选择任务")
    parser.add_argument("--config", type=str, default=None, help="可选：自定义配置路径")
    args = parser.parse_args()

    if args.mode == "train":
        run_script("run_all.py")

    elif args.mode == "predict":
        if args.config:
            run_script("predict.py", ["--config", args.config])
        else:
            run_script("predict.py")

    elif args.mode == "search":
        run_script("hparam_search.py")

    elif args.mode == "update":
        run_script("update_best_config.py")

    elif args.mode == "all":
        run_script("hparam_search.py")
        run_script("update_best_config.py")
        run_script("run_all.py")

if __name__ == "__main__":
    main()