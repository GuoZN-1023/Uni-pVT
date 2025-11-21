# utils/dataset.py
import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_subset(df: pd.DataFrame, subset_cfg: dict | None,
                 expert_col: str | None) -> pd.DataFrame:
    """
    根据 config['subset'] 在 DataFrame 层面筛选数据：
      - include_expert_ids / exclude_expert_ids
      - fraction / max_samples
    """
    if not subset_cfg or not subset_cfg.get("enabled", False):
        return df

    mask = pd.Series(True, index=df.index)

    # 按 no 等编号筛选
    if expert_col is not None and expert_col in df.columns:
        if "include_expert_ids" in subset_cfg and subset_cfg["include_expert_ids"]:
            inc = subset_cfg["include_expert_ids"]
            mask &= df[expert_col].isin(inc)
        if "exclude_expert_ids" in subset_cfg and subset_cfg["exclude_expert_ids"]:
            exc = subset_cfg["exclude_expert_ids"]
            mask &= ~df[expert_col].isin(exc)

    df_sub = df[mask].copy()

    # 随机抽样
    seed = int(subset_cfg.get("seed", 42))
    frac = subset_cfg.get("fraction", None)
    max_samples = subset_cfg.get("max_samples", None)

    if frac is not None:
        df_sub = df_sub.sample(frac=float(frac), random_state=seed)

    if max_samples is not None:
        max_samples = int(max_samples)
        if len(df_sub) > max_samples:
            df_sub = df_sub.sample(n=max_samples, random_state=seed)

    return df_sub.reset_index(drop=True)


class ZDataset(Dataset):
    """
    通用 Z 数据集：
      - 从 csv_path 读取数据
      - target_col 作为 y，其余数值列作为特征
      - expert_col (如 'no') 作为编号，只用于专家预训练，不进入特征
      - subset_cfg 控制子集抽样
      - __getitem__ 返回 (X, y, expert_id)
    """
    def __init__(
        self,
        csv_path: str,
        scaler_path: str,
        train: bool = True,
        target_col: str = "Z (-)",
        expert_col: str | None = "no",
        subset_cfg: dict | None = None,
    ):
        super().__init__()

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if target_col not in df.columns:
            raise ValueError(
                f"target_col='{target_col}' not in columns: {list(df.columns)}"
            )

        # 先按 subset 规则筛数据
        df = apply_subset(df, subset_cfg, expert_col)

        # 只保留数值列（防止字符串）
        num_df = df.select_dtypes(include=[np.number])

        if target_col not in num_df.columns:
            raise ValueError(f"target_col '{target_col}' must be numeric")

        # 目标值 y
        self.y = torch.tensor(
            num_df[target_col].values, dtype=torch.float32
        ).view(-1, 1)

        # 专家编号（如 1/2/3/4），仅供预训练硬分区使用
        self.expert_col = expert_col
        if expert_col is not None and expert_col in num_df.columns:
            self.expert_ids = num_df[expert_col].astype(int).values
        else:
            self.expert_ids = np.full(len(num_df), -1, dtype=int)

        # 特征列：去掉 target 和 expert_col
        drop_cols = [target_col]
        if expert_col is not None and expert_col in num_df.columns:
            drop_cols.append(expert_col)
        feature_cols = [c for c in num_df.columns if c not in drop_cols]
        if len(feature_cols) == 0:
            raise ValueError("No feature columns left after dropping target and expert_col.")

        self.feature_cols = feature_cols
        X_raw = num_df[feature_cols].values.astype(np.float32)

        # 标准化
        if train:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)
        else:
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    f"Scaler not found at {scaler_path}. Run training once to create the scaler."
                )
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X_raw)

        self.X = torch.tensor(X_scaled, dtype=torch.float32)

    @property
    def input_dim(self) -> int:
        return self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        expert_id = int(self.expert_ids[idx])
        return x, y, expert_id


def get_dataloaders(cfg: dict):
    """
    生成 train/val/test 三个 DataLoader。
    会自动应用 cfg['subset'] 的筛选规则，并写回 cfg['model']['input_dim']。
    """
    set_seed(42)

    data_path = cfg["paths"]["data"]
    scaler_path = cfg["paths"]["scaler"]

    target_col = cfg.get("target_col", "Z (-)")
    expert_col = cfg.get("expert_col", "no")
    subset_cfg = cfg.get("subset", None)

    full_dataset = ZDataset(
        csv_path=data_path,
        scaler_path=scaler_path,
        train=True,
        target_col=target_col,
        expert_col=expert_col,
        subset_cfg=subset_cfg,
    )

    cfg.setdefault("model", {})
    cfg["model"]["input_dim"] = full_dataset.input_dim

    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    g = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test], generator=g
    )

    bs = cfg["training"]["batch_size"]
    return {
        "train": DataLoader(train_set, batch_size=bs, shuffle=True),
        "val": DataLoader(val_set, batch_size=bs, shuffle=False),
        "test": DataLoader(test_set, batch_size=bs, shuffle=False),
    }
