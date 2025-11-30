# pt_viz.py
# ------------------------------------------------------------
# chem_viz 风格的 PT 可视化：
# 1) True vs Pred：随 P、T 的 3D 分布 + 各面投影
# 2) Diff (Pred-True 或 |Pred-True|)：随 P、T 的 3D 分布 + 各面投影
#
# 重点优化：PT 平面投影的等高线/填充更“干净”
# - 强制 P=p_r*p_c, T=T_r*T_c（优先级最高）
# - (T, log10(P)) 空间 IDW 插值
# - 高斯平滑抑制波纹
# - 先填充(无边线) + 再叠少量细线(不抢戏)
# ------------------------------------------------------------

import os
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


# 与 chem_viz 保持一致（你也可以按自己的 chem_viz 配色微调）
REGION_COLORS = {1: "#E699A7", 2: "#FEDD9E", 3: "#A6D9C0", 4: "#71A7D2"}
REGION_LABELS = {1: "气相区 (Gas)", 2: "液相区 (Liquid)", 3: "相变区 (Phase Change)", 4: "临界区 (Critical)"}


# ------------------------------ 基础工具 ------------------------------

def re_split_non_alnum(s: str) -> List[str]:
    out, buf = [], []
    for ch in s:
        if ch.isalnum() or ch == "_":
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # fuzzy: contains all tokens
    for cand in candidates:
        toks = [t for t in re_split_non_alnum(cand.lower()) if t]
        for c in df.columns:
            cl = c.lower()
            if all(t in cl for t in toks):
                return c
    return None


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def lighten(hex_color: str, amount: float) -> str:
    """amount in [0,1]: 0 -> original, 1 -> white"""
    r, g, b = hex_to_rgb(hex_color)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return rgb_to_hex((r, g, b))


def make_colorscale(base_hex: str) -> List[List]:
    # light -> dark
    return [
        [0.0, lighten(base_hex, 0.86)],
        [0.35, lighten(base_hex, 0.62)],
        [0.70, lighten(base_hex, 0.34)],
        [1.0, base_hex],
    ]


def normalize_to_opacity(vals: np.ndarray, min_op=0.22, max_op=0.92) -> np.ndarray:
    vals = np.asarray(vals, float)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or abs(vmax - vmin) < 1e-12:
        return np.full_like(vals, 0.80, dtype=float)
    norm = (vals - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    return min_op + (max_op - min_op) * norm


# ------------------------------ P/T 计算（强制） ------------------------------

def compute_PT(df: pd.DataFrame) -> pd.DataFrame:
    """
    强制优先：P = p_r * p_c, T = T_r * T_c
    只有当找不到 pr/pc/Tr/Tc 时，才回退使用已有 P/T 列。
    同时对 p_c 的单位做温和启发式修正（bar->Pa / kPa->Pa）。
    """
    df = df.copy()

    pr = _find_col(df, ["p_r", "pr", "p_r (-)", "p_r(-)", "p reduced", "p_reduced"])
    pc = _find_col(df, ["p_c", "pc", "p_critical", "p critical", "pcrit", "p_crit"])
    Tr = _find_col(df, ["T_r", "tr", "t_r (-)", "t reduced", "t_reduced", "t_r(-)"])
    Tc = _find_col(df, ["T_c", "tc", "T_critical", "t critical", "tcrit", "t_crit"])

    if pr and pc and Tr and Tc:
        prv = df[pr].astype(float).to_numpy()
        pcv = df[pc].astype(float).to_numpy()
        Trv = df[Tr].astype(float).to_numpy()
        Tcv = df[Tc].astype(float).to_numpy()

        # p_c 单位启发式修正（避免把 bar/MPa 当 Pa）
        # 典型：Pa ~ 1e6~1e7；bar ~ 10~100；kPa ~ 1e3~1e4
        med_pc = float(np.nanmedian(pcv))
        if np.isfinite(med_pc) and med_pc < 1e3:
            if med_pc < 200:      # 更像 bar
                pcv = pcv * 1e5
            else:                  # 更像 kPa
                pcv = pcv * 1e3

        df["P"] = prv * pcv
        df["T"] = Trv * Tcv
        return df

    # fallback: already has P/T
    P_col = _find_col(df, ["P", "pressure", "p"])
    T_col = _find_col(df, ["T", "temperature", "temp"])
    if P_col and T_col:
        df["P"] = df[P_col].astype(float)
        df["T"] = df[T_col].astype(float)
        return df

    raise ValueError("找不到用于计算 P、T 的列：需要 (p_r,p_c,T_r,T_c) 或已包含 (P,T)。")


# ------------------------------ 列识别 ------------------------------

def infer_columns(df: pd.DataFrame, prop: Optional[str] = None) -> Dict[str, str]:
    rid = _find_col(df, ["no", "expert_id", "region", "region_id", "phase", "expert"])
    if rid is None:
        raise ValueError("找不到区域/专家编号列（no 或 expert_id）。")

    y_true = _find_col(df, ["y_true", "true", "z_true", "Z_true", "phi_true"])
    y_pred = _find_col(df, ["y_pred", "pred", "z_pred", "Z_pred", "phi_pred", "yhat"])

    if prop and prop in df.columns:
        y_true = prop
        for c in [f"{prop}_pred", f"pred_{prop}", f"{prop}_y_pred"]:
            if c in df.columns:
                y_pred = c
                break

    if y_true is None or y_pred is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        ignore = set([rid, "P", "T"])
        cand = [c for c in num_cols if c not in ignore]
        if len(cand) >= 2:
            y_true, y_pred = cand[0], cand[1]
        else:
            raise ValueError("找不到真实值/预测值列（y_true/y_pred）。")

    return {"region": rid, "y_true": y_true, "y_pred": y_pred}


# ------------------------------ 插值 + 平滑 ------------------------------

def idw_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.1,
    eps: float = 1e-12,
    max_norm_dist: float = 0.16,
) -> np.ndarray:
    """
    简单 IDW 插值，并用“最近样本距离阈值”掩膜避免过远外推。
    建议在 (T, log10(P)) 空间做插值。
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    gx = np.asarray(grid_x, float)
    gy = np.asarray(grid_y, float)

    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    dx = max(xmax - xmin, 1e-12)
    dy = max(ymax - ymin, 1e-12)

    xn = (x - xmin) / dx
    yn = (y - ymin) / dy
    gxn = (gx - xmin) / dx
    gyn = (gy - ymin) / dy

    M = gxn.size
    N = xn.size

    gxn2 = gxn.reshape(M, 1)
    gyn2 = gyn.reshape(M, 1)
    d2 = (gxn2 - xn.reshape(1, N)) ** 2 + (gyn2 - yn.reshape(1, N)) ** 2
    d = np.sqrt(d2) + eps

    w = 1.0 / (d ** power)
    zhat = (w @ z.reshape(N, 1))[:, 0] / (np.sum(w, axis=1) + eps)

    dmin = np.min(d, axis=1)
    zhat = np.where(dmin <= max_norm_dist, zhat, np.nan)

    return zhat.reshape(gx.shape)


def _gaussian_smooth_2d(z: np.ndarray, sigma: float = 1.1) -> np.ndarray:
    """
    不依赖 scipy 的 2D 高斯平滑（separable 1D kernel）。
    会先用插值方式填补 NaN 再平滑，最后恢复 NaN 区域。
    """
    if sigma <= 0 or z is None:
        return z
    z = np.array(z, float)
    mask = np.isfinite(z)
    if mask.sum() < 12:
        return z

    radius = int(max(1, round(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(xs**2) / (2 * sigma**2))
    k /= k.sum()

    def conv_axis(a: np.ndarray, axis: int) -> np.ndarray:
        a = np.swapaxes(a, axis, 0)
        out = np.empty_like(a)
        for j in range(a.shape[1]):
            v = a[:, j]
            vv = v.copy()
            bad = ~np.isfinite(vv)
            if bad.any():
                good = np.where(~bad)[0]
                if len(good) == 0:
                    out[:, j] = v
                    continue
                vv[bad] = np.interp(np.where(bad)[0], good, vv[good])
            out[:, j] = np.convolve(vv, k, mode="same")
        out = np.swapaxes(out, 0, axis)
        return out

    z2 = z.copy()
    z2 = conv_axis(z2, axis=0)
    z2 = conv_axis(z2, axis=1)
    z2[~mask] = np.nan
    return z2


# ------------------------------ 绘图：3D ------------------------------

def make_3d_true_pred(
    df: pd.DataFrame,
    col_true: str,
    col_pred: str,
    col_region: str,
    title: str,
    value_label: str,
) -> go.Figure:
    traces = []
    for r in [1, 2, 3, 4]:
        sub = df[df[col_region] == r]
        if len(sub) == 0:
            continue
        base = REGION_COLORS.get(r, "#999999")
        label = REGION_LABELS.get(r, f"Region {r}")

        traces.append(
            go.Scatter3d(
                x=sub["P"], y=sub["T"], z=sub[col_true],
                mode="markers",
                name=f"{label} · True",
                marker=dict(size=3, color=base, opacity=0.88),
                hovertemplate=f"<b>{label}</b><br>P=%{{x:.2e}} Pa<br>T=%{{y:.2f}} K<br>True={value_label}: %{{z:.6g}}<extra></extra>",
            )
        )
        traces.append(
            go.Scatter3d(
                x=sub["P"], y=sub["T"], z=sub[col_pred],
                mode="markers",
                name=f"{label} · Pred",
                marker=dict(size=3, color=base, opacity=0.45, symbol="diamond"),
                hovertemplate=f"<b>{label}</b><br>P=%{{x:.2e}} Pa<br>T=%{{y:.2f}} K<br>Pred={value_label}: %{{z:.6g}}<extra></extra>",
            )
        )
    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="压力 P (Pa)"),
            yaxis=dict(title="温度 T (K)"),
            zaxis=dict(title=value_label),
        ),
        margin=dict(l=0, r=0, b=0, t=42),
        legend=dict(x=0, y=1),
        uirevision="true",
    )
    return fig


def make_3d_single(
    df: pd.DataFrame,
    z_col: str,
    col_region: str,
    title: str,
    z_label: str,
) -> go.Figure:
    traces = []
    for r in [1, 2, 3, 4]:
        sub = df[df[col_region] == r]
        if len(sub) == 0:
            continue
        base = REGION_COLORS.get(r, "#999999")
        label = REGION_LABELS.get(r, f"Region {r}")
        traces.append(
            go.Scatter3d(
                x=sub["P"], y=sub["T"], z=sub[z_col],
                mode="markers",
                name=label,
                marker=dict(size=3, color=base, opacity=0.85),
                hovertemplate=f"<b>{label}</b><br>P=%{{x:.2e}} Pa<br>T=%{{y:.2f}} K<br>{z_label}: %{{z:.6g}}<extra></extra>",
            )
        )
    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="压力 P (Pa)"),
            yaxis=dict(title="温度 T (K)"),
            zaxis=dict(title=z_label),
        ),
        margin=dict(l=0, r=0, b=0, t=42),
        legend=dict(x=0, y=1),
        uirevision="true",
    )
    return fig


# ------------------------------ 绘图：投影 ------------------------------

def make_projection_xz(
    df: pd.DataFrame,
    x_col: str,
    z_col: str,
    col_region: str,
    title: str,
    x_label: str,
    z_label: str,
    x_log: bool = False,
) -> go.Figure:
    traces = []
    for r in [1, 2, 3, 4]:
        sub = df[df[col_region] == r]
        if len(sub) == 0:
            continue
        base = REGION_COLORS.get(r, "#999999")
        label = REGION_LABELS.get(r, f"Region {r}")
        z = sub[z_col].to_numpy(float)
        op = normalize_to_opacity(z)
        traces.append(
            go.Scattergl(
                x=sub[x_col],
                y=sub[z_col],
                mode="markers",
                name=label,
                marker=dict(size=6, color=base, opacity=op),
                hovertemplate=f"<b>{label}</b><br>{x_label}=%{{x}}<br>{z_label}=%{{y:.6g}}<extra></extra>",
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_label, type=("log" if x_log else "linear")),
        yaxis=dict(title=z_label),
        hovermode="closest",
        margin=dict(l=70, r=20, b=60, t=45),
        uirevision="true",
    )
    return fig


def make_projection_pt_with_contours(
    df: pd.DataFrame,
    value_col: str,
    col_region: str,
    title: str,
    overlay_points: bool,
    grid_n: int = 90,
    yaxis_log: bool = True,
) -> go.Figure:
    """
    PT 平面投影（优化版）：
    - (T, log10(P)) 空间 IDW
    - 高斯平滑抑制波纹
    - 先“填充无边线”再“少量细线”叠加，整体更干净
    """
    traces: List[go.BaseTraceType] = []
    LINE_COLOR = "rgba(20,20,20,0.55)"

    for r in [1, 2, 3, 4]:
        sub = df[df[col_region] == r]
        if len(sub) < 12:
            continue

        base = REGION_COLORS.get(r, "#999999")
        label = REGION_LABELS.get(r, f"Region {r}")

        T = sub["T"].to_numpy(float)
        P = sub["P"].to_numpy(float)
        z = sub[value_col].to_numpy(float)

        logP = np.log10(np.clip(P, 1e-30, None))

        tmin, tmax = float(np.nanmin(T)), float(np.nanmax(T))
        pmin, pmax = float(np.nanmin(logP)), float(np.nanmax(logP))
        if not (np.isfinite(tmin) and np.isfinite(tmax) and np.isfinite(pmin) and np.isfinite(pmax)):
            continue

        pad_t = 0.04 * (tmax - tmin + 1e-12)
        pad_p = 0.04 * (pmax - pmin + 1e-12)

        gt = np.linspace(tmin - pad_t, tmax + pad_t, grid_n)
        gp = np.linspace(pmin - pad_p, pmax + pad_p, grid_n)
        GT, GP = np.meshgrid(gt, gp)

        ZG = idw_grid(T, logP, z, GT, GP, power=2.1, max_norm_dist=0.16)
        ZG = _gaussian_smooth_2d(ZG, sigma=1.1)

        # 1) 填充（不画线）
        traces.append(
            go.Contour(
                x=gt,
                y=10 ** gp,
                z=ZG,
                name=f"{label} · fill",
                colorscale=make_colorscale(base),
                opacity=0.92,
                showscale=False,
                contours=dict(coloring="heatmap", showlines=False),
                hovertemplate=f"<b>{label}</b><br>T=%{{x:.2f}} K<br>P=%{{y:.2e}} Pa<br>≈%{{z:.6g}}<extra></extra>",
            )
        )

        # 2) 少量线条（注意：ncontours 必须在顶层！）
        traces.append(
            go.Contour(
                x=gt,
                y=10 ** gp,
                z=ZG,
                name=f"{label} · lines",
                showscale=False,
                ncontours=7,  # ✅ 顶层参数，不要放进 contours={}
                contours=dict(coloring="none", showlines=True),
                line=dict(width=0.8, smoothing=1.0, color=LINE_COLOR),
                opacity=1.0,
                hoverinfo="skip",
            )
        )

        # 3) 叠加散点
        if overlay_points:
            op = normalize_to_opacity(z, min_op=0.25, max_op=0.90)
            traces.append(
                go.Scattergl(
                    x=T,
                    y=P,
                    mode="markers",
                    name=f"{label} · pts",
                    marker=dict(size=5, color=base, opacity=op),
                    text=[f"{v:.6g}" for v in z],
                    hovertemplate=f"<b>{label}</b><br>T=%{{x:.2f}} K<br>P=%{{y:.2e}} Pa<br>value=%{{text}}<extra></extra>",
                )
            )

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        xaxis=dict(title="温度 T (K)"),
        yaxis=dict(title="压力 P (Pa)", type=("log" if yaxis_log else "linear")),
        hovermode="closest",
        margin=dict(l=70, r=20, b=60, t=45),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        uirevision="true",
    )
    return fig


# ------------------------------ Dashboard HTML ------------------------------

def build_dashboard_html(blocks: List[Tuple[str, go.Figure]], out_path: str, page_title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    divs = []
    for i, (title, fig) in enumerate(blocks):
        div = pio.to_html(fig, include_plotlyjs=("cdn" if i == 0 else False), full_html=False)
        divs.append(f"<h2 style='font-family:system-ui;margin:18px 0 10px'>{title}</h2>\n{div}")
    html = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{page_title}</title>
<style>
  body {{ margin: 0; padding: 18px 22px; background: #0b0f14; color: #e9eef6; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }}
  h1 {{ margin: 6px 0 16px; font-size: 20px; }}
  h2 {{ font-size: 15px; font-weight: 650; color: #e9eef6; }}
  .hint {{ color:#9db0c6; font-size: 12px; margin-bottom: 12px; line-height: 1.5; }}
  .card {{ background: #111826; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 12px 12px 6px; margin: 14px 0; }}
</style>
</head>
<body>
  <h1>{page_title}</h1>
  <div class="hint">
    PT 投影：每个区域用固定基色（与 chem_viz 一致），以“颜色深浅”表示等高填充；再叠少量细线用于读数。<br/>
    P 轴默认对数坐标；P、T 默认由 p=p_r·p_c，T=T_r·T_c 计算（若你想强制使用已有 P/T，可自行改 compute_PT 的优先级）。
  </div>
  {''.join([f"<div class='card'>{d}</div>" for d in divs])}
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ------------------------------ 可选：从 checkpoint 直接生成结果表 ------------------------------

def load_from_checkpoint(config_path: str, save_dir: str, split: str = "test") -> pd.DataFrame:
    """
    不改动原训练/预测代码的前提下，从 config + best_model 生成可视化所需数据表：
    [num_df columns] + y_true + y_pred + expert_id
    """
    import yaml
    import torch
    from torch.utils.data import DataLoader, random_split
    from models.fusion_model import FusionModel
    from utils.dataset import ZDataset

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = ZDataset(cfg, train=False)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    g = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)

    if split == "train":
        subset = train_set
    elif split == "val":
        subset = val_set
    else:
        subset = test_set

    idx = np.array(subset.indices, dtype=int)
    df_states = dataset.num_df.iloc[idx].reset_index(drop=True)
    expert_id = dataset.expert_ids[idx].astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(cfg).to(device)
    ckpt = os.path.join(save_dir, "checkpoints", "best_model.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    loader = DataLoader(subset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(device)
            y = y.to(device)
            pred, _, _ = model(x)
            y_true_list.append(y.detach().cpu().numpy().reshape(-1))
            y_pred_list.append(pred.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    df_out = df_states.copy()
    df_out["y_true"] = y_true
    df_out["y_pred"] = y_pred
    df_out["expert_id"] = expert_id
    return df_out


# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="优先：包含特征+真实/预测的 CSV（例如 eval/test_predictions.csv）")
    ap.add_argument("--config", default=None, help="若不使用 CSV，可给 config.yaml 并用 checkpoint 生成结果表")
    ap.add_argument("--save_dir", default=None, help="训练输出目录（包含 checkpoints/best_model.pt）")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="checkpoint 模式下选择哪一段数据")
    ap.add_argument("--outdir", default=None, help="输出目录。默认与 CSV 同级创建 pt_viz/ 或 save_dir/pt_viz/")
    ap.add_argument("--prop", default=None, help="可选：目标物性名（辅助识别列）")
    ap.add_argument("--diff_mode", default="pred_minus_true", choices=["pred_minus_true", "abs"], help="差值定义")
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        base_for_out = os.path.dirname(os.path.abspath(args.csv))
    else:
        if not (args.config and args.save_dir):
            raise ValueError("要么传 --csv，要么传 --config + --save_dir（从 checkpoint 直接生成结果表）。")
        df = load_from_checkpoint(args.config, args.save_dir, split=args.split)
        base_for_out = os.path.abspath(args.save_dir)

    df = compute_PT(df)
    cols = infer_columns(df, prop=args.prop)
    col_r = cols["region"]
    col_true = cols["y_true"]
    col_pred = cols["y_pred"]

    df[col_r] = df[col_r].astype(int)

    outdir = args.outdir or os.path.join(base_for_out, "pt_viz")
    os.makedirs(outdir, exist_ok=True)

    value_label = args.prop or "Value"

    # --- True vs Pred ---
    fig_3d = make_3d_true_pred(df, col_true, col_pred, col_r, "原始值 vs 预测值：随 P、T 变化的 3D 分布", value_label)
    fig_pt_pred = make_projection_pt_with_contours(df, col_pred, col_r, "P–T 平面投影（Pred 等高线/填充）", overlay_points=True)
    fig_t_true = make_projection_xz(df, "T", col_true, col_r, "面投影：T–True", "T (K)", f"True {value_label}")
    fig_t_pred = make_projection_xz(df, "T", col_pred, col_r, "面投影：T–Pred", "T (K)", f"Pred {value_label}")
    fig_p_true = make_projection_xz(df, "P", col_true, col_r, "面投影：P–True（logP）", "P (Pa)", f"True {value_label}", x_log=True)
    fig_p_pred = make_projection_xz(df, "P", col_pred, col_r, "面投影：P–Pred（logP）", "P (Pa)", f"Pred {value_label}", x_log=True)

    # --- Diff ---
    df2 = df.copy()
    if args.diff_mode == "abs":
        df2["diff"] = np.abs(df2[col_pred].astype(float) - df2[col_true].astype(float))
        diff_label = f"|Pred - True| ({value_label})"
    else:
        df2["diff"] = df2[col_pred].astype(float) - df2[col_true].astype(float)
        diff_label = f"Pred - True ({value_label})"

    fig_3d_diff = make_3d_single(df2, "diff", col_r, "差值：随 P、T 变化的 3D 分布", diff_label)
    fig_pt_diff = make_projection_pt_with_contours(df2, "diff", col_r, "P–T 平面投影（差值等高线/填充）", overlay_points=True)
    fig_t_diff = make_projection_xz(df2, "T", "diff", col_r, "面投影：T–差值", "T (K)", diff_label)
    fig_p_diff = make_projection_xz(df2, "P", "diff", col_r, "面投影：P–差值（logP）", "P (Pa)", diff_label, x_log=True)

    dashboard = [
        ("3D：True vs Pred", fig_3d),
        ("PT：Pred 等高线/填充（优化后）", fig_pt_pred),
        ("面投影：T–True", fig_t_true),
        ("面投影：T–Pred", fig_t_pred),
        ("面投影：P–True（logP）", fig_p_true),
        ("面投影：P–Pred（logP）", fig_p_pred),
        ("3D：差值", fig_3d_diff),
        ("PT：差值等高线/填充（优化后）", fig_pt_diff),
        ("面投影：T–差值", fig_t_diff),
        ("面投影：P–差值（logP）", fig_p_diff),
    ]

    out_html = os.path.join(outdir, "pt_viz_dashboard.html")
    build_dashboard_html(dashboard, out_html, "PT 三维/投影可视化（chem_viz 风格 · 等高线优化版）")
    print(f"[OK] Wrote: {out_html}")


if __name__ == "__main__":
    main()
