# utils/trainer.py
import os
import inspect
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _set_requires_grad(params, flag: bool):
    for p in params:
        p.requires_grad = flag


def _filter_trainable(params):
    return [p for p in params if getattr(p, "requires_grad", False)]


def _apply_temperature_to_weights(w: torch.Tensor, tau: float) -> torch.Tensor:
    """
    w: (B,4) probability weights (softmax output)
    tau: temperature; <1 => sharper, >1 => smoother
    Using: softmax(log(w)/tau) == normalize(w^(1/tau))
    """
    tau = float(max(tau, 1e-6))
    w_pow = torch.pow(torch.clamp(w, 1e-12, 1.0), 1.0 / tau)
    w_tau = w_pow / torch.clamp(w_pow.sum(dim=1, keepdim=True), 1e-12)
    return w_tau


def _temperature_schedule(cfg, stage: str, epoch: int, total_epochs: int) -> float:
    """
    stage: "stage2" or "finetune"

    Priority:
      1) training.temperature_schedule.<stage>.{start,end,mode}
      2) training.gate_temp_start / gate_temp_end / gate_temp_mode  (兼容你当前 config)
      3) default 1.0
    """
    training = cfg.get("training", {}) or {}

    ts = training.get("temperature_schedule", {}) or {}
    st = ts.get(stage, None)

    if st is not None:
        start = float(st.get("start", 1.0))
        end = float(st.get("end", 1.0))
        mode = str(st.get("mode", "linear")).lower()
    else:
        start = float(training.get("gate_temp_start", 1.0))
        end = float(training.get("gate_temp_end", 1.0))
        mode = str(training.get("gate_temp_mode", "linear")).lower()

    if total_epochs <= 1:
        return float(end)

    t = (epoch - 1) / (total_epochs - 1)

    if mode in ["linear", "lin"]:
        tau = start + (end - start) * t
    elif mode in ["exp", "exponential"]:
        s = max(start, 1e-8)
        e = max(end, 1e-8)
        tau = s * ((e / s) ** t)
    else:
        tau = start

    return float(tau)


def _build_lr_scheduler(cfg, optimizer, stage: str, total_epochs: int, steps_per_epoch: int):
    """
    Returns (scheduler_or_None, interval_str) where interval_str in {"epoch","batch"}.

    Supported:
      - cosine
      - step
      - exponential
      - reduce_on_plateau  (兼容 torch 版本：verbose 参数自动检测)
      - onecycle
    """
    training_cfg = cfg.get("training", {}) or {}
    sched_cfg = (training_cfg.get("lr_schedule", {}) or {})
    stage_cfg = sched_cfg.get(stage, None)
    if not stage_cfg:
        return None, "epoch"

    name = str(stage_cfg.get("name", "")).lower().strip()
    interval = str(stage_cfg.get("interval", "epoch")).lower().strip()

    if name in ["none", "off", ""]:
        return None, interval

    if name in ["cosine", "cosineannealing", "cosineannealinglr"]:
        eta_min = float(stage_cfg.get("eta_min", 1e-6))
        t_max = int(stage_cfg.get("t_max", total_epochs))
        t_max = max(t_max, 1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        ), interval

    if name in ["step", "steplr"]:
        step_size = int(stage_cfg.get("step_size", max(total_epochs // 3, 1)))
        gamma = float(stage_cfg.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        ), interval

    if name in ["exp", "exponential", "exponentiallr"]:
        gamma = float(stage_cfg.get("gamma", 0.98))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma), interval

    if name in ["reduce_on_plateau", "plateau", "reducelronplateau"]:
        factor = float(stage_cfg.get("factor", 0.5))
        patience = int(stage_cfg.get("patience", 10))
        min_lr = float(stage_cfg.get("min_lr", 1e-6))
        mode = str(stage_cfg.get("mode", "min"))
        threshold = float(stage_cfg.get("threshold", 1e-4))
        cooldown = int(stage_cfg.get("cooldown", 0))
        eps = float(stage_cfg.get("eps", 1e-8))

        kwargs = dict(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )
        sig = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__)
        if "verbose" in sig.parameters:
            kwargs["verbose"] = bool(stage_cfg.get("verbose", True))

        return torch.optim.lr_scheduler.ReduceLROnPlateau(**kwargs), interval

    if name in ["onecycle", "onecyclelr"]:
        interval = "batch"
        max_lr = stage_cfg.get("max_lr", None)
        if max_lr is None:
            max_lr = optimizer.param_groups[0]["lr"]
        max_lr = float(max_lr)

        pct_start = float(stage_cfg.get("pct_start", 0.3))
        div_factor = float(stage_cfg.get("div_factor", 25.0))
        final_div_factor = float(stage_cfg.get("final_div_factor", 1e4))
        anneal_strategy = str(stage_cfg.get("anneal_strategy", "cos"))

        total_steps = int(total_epochs * max(steps_per_epoch, 1))
        total_steps = max(total_steps, 1)

        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy=anneal_strategy,
        ), interval

    return None, interval


def _is_plateau_scheduler(scheduler) -> bool:
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def _step_scheduler(scheduler, interval: str, when: str, metric: float = None):
    """
    when: "batch" or "epoch_end"
    interval: "batch" or "epoch"
    """
    if scheduler is None:
        return
    interval = (interval or "epoch").lower().strip()
    if interval == "batch" and when == "batch":
        scheduler.step()
        return
    if interval == "epoch" and when == "epoch_end":
        if _is_plateau_scheduler(scheduler):
            if metric is None:
                return
            scheduler.step(metric)
        else:
            scheduler.step()


def _eval_pretrain_expert_metrics(model, val_loader, device):
    """
    Pretrain评估：
      - 每个专家在自己子集(expert_id==k)上的 MAE/MSE/R2
      - overall：按 expert_id 硬路由选对应专家输出，在全验证集上算一次
    """
    model.eval()
    buf = {1: {"y": [], "p": []}, 2: {"y": [], "p": []}, 3: {"y": [], "p": []}, 4: {"y": [], "p": []}}
    overall_y, overall_p = [], []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) != 3:
                continue
            x, y, expert_ids = batch
            x = x.to(device)
            y = y.to(device)
            expert_ids = expert_ids.to(device).long()

            out = model(x)
            if isinstance(out, (list, tuple)) and len(out) >= 3:
                expert_outputs = out[2]
            elif isinstance(out, dict) and "expert_outputs" in out:
                expert_outputs = out["expert_outputs"]
            else:
                raise RuntimeError("Model forward should return expert_outputs for pretrain metrics.")

            for k in [1, 2, 3, 4]:
                mask = (expert_ids == k)
                if mask.any():
                    yk = y[mask].detach().cpu().numpy().reshape(-1)
                    pk = expert_outputs[mask, k - 1].detach().cpu().numpy().reshape(-1)
                    buf[k]["y"].append(yk)
                    buf[k]["p"].append(pk)

            idx = (expert_ids - 1).view(-1, 1)
            sel = expert_outputs.gather(1, idx).view(-1)
            overall_y.append(y.detach().cpu().numpy().reshape(-1))
            overall_p.append(sel.detach().cpu().numpy().reshape(-1))

    rows = []
    for k in [1, 2, 3, 4]:
        if len(buf[k]["y"]) == 0:
            rows.append({"expert": str(k), "n": 0, "mae": np.nan, "mse": np.nan, "r2": np.nan})
            continue
        yt = np.concatenate(buf[k]["y"], axis=0)
        yp = np.concatenate(buf[k]["p"], axis=0)
        rows.append({
            "expert": str(k),
            "n": int(yt.shape[0]),
            "mae": float(mean_absolute_error(yt, yp)),
            "mse": float(mean_squared_error(yt, yp)),
            "r2": float(r2_score(yt, yp)) if yt.shape[0] >= 2 else np.nan,
        })

    if len(overall_y) == 0:
        rows.append({"expert": "hard_routed_overall", "n": 0, "mae": np.nan, "mse": np.nan, "r2": np.nan})
    else:
        yt = np.concatenate(overall_y, axis=0)
        yp = np.concatenate(overall_p, axis=0)
        rows.append({
            "expert": "hard_routed_overall",
            "n": int(yt.shape[0]),
            "mae": float(mean_absolute_error(yt, yp)),
            "mse": float(mean_squared_error(yt, yp)),
            "r2": float(r2_score(yt, yp)) if yt.shape[0] >= 2 else np.nan,
        })

    return rows


def _make_optimizer(kind: str, params, lr: float, betas=(0.9, 0.999), weight_decay: float = 0.0):
    kind = (kind or "adam").lower().strip()
    lr = float(lr)
    weight_decay = float(weight_decay)

    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if kind == "adam":
        # torch Adam 的 weight_decay 是 L2（不是 decoupled），但依然可用；这里保持为 0 更保险
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if kind == "sgd":
        momentum = betas[0] if betas else 0.9
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    # fallback
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


def train_model(model, dataloaders, criterion, optimizer, cfg, device, logger):
    """
    Stage 1: pretrain experts (hard routing by expert_id / 'no')
    Stage 2: train gate + MoE (experts frozen)  -> 默认 AdamW + gate_weight_decay
    Stage 3: optional finetune (unfreeze gate + selected experts) -> 默认 AdamW + finetune_weight_decay
    """
    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    steps_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else 1

    total_epochs = int(cfg["training"]["epochs"])
    pretrain_epochs = int(cfg["training"].get("pretrain_epochs", 0))
    patience = int(cfg["training"]["early_stopping_patience"])

    # collect params
    expert_params = []
    if hasattr(model, "expert_gas"):
        expert_params += list(model.expert_gas.parameters())
    if hasattr(model, "expert_liq"):
        expert_params += list(model.expert_liq.parameters())
    if hasattr(model, "expert_crit"):
        expert_params += list(model.expert_crit.parameters())
    if hasattr(model, "expert_extra"):
        expert_params += list(model.expert_extra.parameters())

    gate_params = list(model.gate.parameters())

    base_lr = float(cfg["training"]["learning_rate"])

    # ---------- Stage1 optimizer (experts) ----------
    pre_opt_kind = str(cfg["training"].get("pretrain_optimizer", "adam")).lower()
    pre_weight_decay = float(cfg["training"].get("pretrain_weight_decay", 0.0))
    optim_experts = None
    if pretrain_epochs > 0:
        optim_experts = _make_optimizer(
            pre_opt_kind, expert_params, lr=base_lr, betas=(0.9, 0.999), weight_decay=pre_weight_decay
        )

    # ---------- Stage2 optimizer (gate) : 必要修改点 ----------
    gate_opt_kind = str(cfg["training"].get("gate_optimizer", "adamw")).lower()
    gate_weight_decay = float(cfg["training"].get("gate_weight_decay", 1e-4))
    gate_betas = cfg["training"].get("gate_betas", [0.9, 0.99])
    gate_betas = (float(gate_betas[0]), float(gate_betas[1])) if isinstance(gate_betas, (list, tuple)) and len(gate_betas) >= 2 else (0.9, 0.99)

    optim_gate = _make_optimizer(
        gate_opt_kind, gate_params, lr=base_lr, betas=gate_betas, weight_decay=gate_weight_decay
    )

    # ---------- schedulers ----------
    sched_pre, sched_pre_interval = (None, "epoch")
    if optim_experts is not None:
        sched_pre, sched_pre_interval = _build_lr_scheduler(cfg, optim_experts, "pretrain", pretrain_epochs, steps_per_epoch)

    sched_gate, sched_gate_interval = _build_lr_scheduler(cfg, optim_gate, "stage2", total_epochs, steps_per_epoch)

    # ---------- records ----------
    train_losses, val_losses = [], []
    mse_list, nonneg_list, smooth_list, entropy_list = [], [], [], []
    gate_e1_list, gate_e2_list, gate_e3_list, gate_e4_list = [], [], [], []
    train_mae_list, val_mae_list = [], []
    train_mse_metric_list, val_mse_metric_list = [], []
    train_r2_list, val_r2_list = [], []

    pretrain_metrics_rows = []

    # ===================== Stage 1: Pretrain experts =====================
    if pretrain_epochs > 0:
        logger.info(f"=== Stage 1: Pretraining experts for {pretrain_epochs} epochs (hard routing by 'no') ===")
        _set_requires_grad(gate_params, False)
        _set_requires_grad(expert_params, True)

        for epoch in range(1, pretrain_epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                if len(batch) != 3:
                    raise RuntimeError("Pretraining requires (x, y, expert_id) in dataset.")
                x, y, expert_ids = batch
                x = x.to(device)
                y = y.to(device)
                expert_ids = expert_ids.to(device).long()

                optim_experts.zero_grad()

                out = model(x)
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    expert_outputs = out[2]
                elif isinstance(out, dict) and "expert_outputs" in out:
                    expert_outputs = out["expert_outputs"]
                else:
                    raise RuntimeError("Model forward should return expert_outputs in pretrain stage.")

                idx = expert_ids - 1
                selected = expert_outputs.gather(1, idx.view(-1, 1))  # [B,1]

                loss, _ = criterion(selected, y, expert_id=expert_ids, gate_w=None, extra=None)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(expert_params, 5.0)
                optim_experts.step()

                _step_scheduler(sched_pre, sched_pre_interval, when="batch")

                running_loss += loss.item()
                n_batches += 1

            avg_loss = running_loss / max(n_batches, 1)
            _step_scheduler(sched_pre, sched_pre_interval, when="epoch_end")

            rows = _eval_pretrain_expert_metrics(model, val_loader, device)
            for r in rows:
                pretrain_metrics_rows.append({
                    "epoch": epoch,
                    "expert": r["expert"],
                    "n": r["n"],
                    "mae": r["mae"],
                    "mse": r["mse"],
                    "r2": r["r2"],
                    "lr": float(optim_experts.param_groups[0]["lr"]),
                    "train_loss": float(avg_loss),
                })

            msg = f"[Pretrain] Epoch {epoch}/{pretrain_epochs} TrainLoss={avg_loss:.6f} LR={optim_experts.param_groups[0]['lr']:.3e} | "
            parts = []
            for r in rows:
                tag = "Overall" if r["expert"] == "hard_routed_overall" else f"E{r['expert']}"
                parts.append(f"{tag}: n={r['n']} MAE={r['mae']:.4g} MSE={r['mse']:.4g} R2={r['r2']:.4g}")
            logger.info(msg + " ; ".join(parts))

        df_pre = pd.DataFrame(pretrain_metrics_rows)
        df_pre.to_csv(f"{save_dir}/plots/pretrain_expert_metrics.csv", index=False)

        _set_requires_grad(expert_params, False)
        _set_requires_grad(gate_params, True)
    else:
        # 如果没有预训练，也建议 Stage2 只训 gate（否则会马上一锅端把专家也训练了，风险大）
        _set_requires_grad(expert_params, False)
        _set_requires_grad(gate_params, True)

    # ===================== Stage 2: Train gate with early stopping =====================
    logger.info(f"=== Stage 2: Training gate + MoE for up to {total_epochs} epochs with early stopping ===")
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, total_epochs + 1):
        tau = _temperature_schedule(cfg, "stage2", epoch, total_epochs)

        model.train()
        running_train_loss = 0.0
        mse_epoch = 0.0
        nonneg_epoch = 0.0
        smooth_epoch = 0.0
        entropy_epoch = 0.0
        n_batches = 0
        w_accum = None

        train_y_true, train_y_pred = [], []

        for batch in train_loader:
            expert_ids = None
            if len(batch) == 3:
                x, y, expert_ids = batch
            elif len(batch) == 2:
                x, y = batch
            else:
                raise RuntimeError("Unexpected batch format in training stage.")
            x = x.to(device)
            y = y.to(device)
            if expert_ids is not None:
                expert_ids = expert_ids.to(device).long()

            optim_gate.zero_grad()

            out = model(x)
            if isinstance(out, (list, tuple)) and len(out) >= 3:
                w_raw, expert_outputs = out[1], out[2]
            elif isinstance(out, dict):
                w_raw = out.get("w_raw", None) or out.get("gate_w", None)
                expert_outputs = out.get("expert_outputs", None)
                if w_raw is None or expert_outputs is None:
                    raise RuntimeError("Model output dict missing gate weights or expert outputs.")
            else:
                raise RuntimeError("Unexpected model forward output format.")

            w_tau = _apply_temperature_to_weights(w_raw, tau)
            preds = (w_tau * expert_outputs).sum(dim=1, keepdim=True)

            loss, loss_dict = criterion(preds, y, expert_id=expert_ids, gate_w=w_tau, extra=None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_params, 5.0)
            optim_gate.step()

            _step_scheduler(sched_gate, sched_gate_interval, when="batch")

            running_train_loss += loss.item()
            mse_epoch += loss_dict.get("MSE", 0.0)
            nonneg_epoch += loss_dict.get("NonNeg", 0.0)
            smooth_epoch += loss_dict.get("Smooth", 0.0)
            entropy_epoch += loss_dict.get("Entropy", 0.0)
            n_batches += 1

            train_y_true.append(y.detach().cpu().numpy())
            train_y_pred.append(preds.detach().cpu().numpy())

            w_mean_batch = w_tau.mean(dim=0)
            w_accum = w_mean_batch.detach() if w_accum is None else (w_accum + w_mean_batch.detach())

        train_loss = running_train_loss / max(n_batches, 1)
        mse_epoch /= max(n_batches, 1)
        nonneg_epoch /= max(n_batches, 1)
        smooth_epoch /= max(n_batches, 1)
        entropy_epoch /= max(n_batches, 1)

        train_y_true_np = np.concatenate(train_y_true, axis=0).reshape(-1)
        train_y_pred_np = np.concatenate(train_y_pred, axis=0).reshape(-1)
        train_mae = mean_absolute_error(train_y_true_np, train_y_pred_np)
        train_mse_metric = mean_squared_error(train_y_true_np, train_y_pred_np)
        train_r2 = r2_score(train_y_true_np, train_y_pred_np)

        # ----- validation -----
        model.eval()
        running_val_loss = 0.0
        n_val_batches = 0
        val_y_true, val_y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                expert_ids = None
                if len(batch) == 3:
                    x, y, expert_ids = batch
                elif len(batch) == 2:
                    x, y = batch
                else:
                    raise RuntimeError("Unexpected batch format in validation stage.")
                x = x.to(device)
                y = y.to(device)
                if expert_ids is not None:
                    expert_ids = expert_ids.to(device).long()

                out = model(x)
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    w_raw, expert_outputs = out[1], out[2]
                elif isinstance(out, dict):
                    w_raw = out.get("w_raw", None) or out.get("gate_w", None)
                    expert_outputs = out.get("expert_outputs", None)
                    if w_raw is None or expert_outputs is None:
                        raise RuntimeError("Model output dict missing gate weights or expert outputs.")
                else:
                    raise RuntimeError("Unexpected model forward output format.")

                w_tau = _apply_temperature_to_weights(w_raw, tau)
                preds = (w_tau * expert_outputs).sum(dim=1, keepdim=True)

                vloss, _ = criterion(preds, y, expert_id=expert_ids, gate_w=w_tau, extra=None)
                running_val_loss += vloss.item()
                n_val_batches += 1

                val_y_true.append(y.detach().cpu().numpy())
                val_y_pred.append(preds.detach().cpu().numpy())

        val_loss = running_val_loss / max(n_val_batches, 1)

        val_y_true_np = np.concatenate(val_y_true, axis=0).reshape(-1)
        val_y_pred_np = np.concatenate(val_y_pred, axis=0).reshape(-1)
        val_mae = mean_absolute_error(val_y_true_np, val_y_pred_np)
        val_mse_metric = mean_squared_error(val_y_true_np, val_y_pred_np)
        val_r2 = r2_score(val_y_true_np, val_y_pred_np)

        _step_scheduler(sched_gate, sched_gate_interval, when="epoch_end", metric=val_loss)

        if w_accum is not None:
            w_mean = (w_accum / max(n_batches, 1)).cpu().numpy()
            gate_e1_list.append(float(w_mean[0]))
            gate_e2_list.append(float(w_mean[1]))
            gate_e3_list.append(float(w_mean[2]))
            gate_e4_list.append(float(w_mean[3]))
        else:
            gate_e1_list.append(0.0)
            gate_e2_list.append(0.0)
            gate_e3_list.append(0.0)
            gate_e4_list.append(0.0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        mse_list.append(mse_epoch)
        nonneg_list.append(nonneg_epoch)
        smooth_list.append(smooth_epoch)
        entropy_list.append(entropy_epoch)
        train_mae_list.append(train_mae)
        val_mae_list.append(val_mae)
        train_mse_metric_list.append(train_mse_metric)
        val_mse_metric_list.append(val_mse_metric)
        train_r2_list.append(train_r2)
        val_r2_list.append(val_r2)

        logger.info(
            f"[Epoch {epoch}/{total_epochs}] tau={tau:.3f} "
            f"LR={optim_gate.param_groups[0]['lr']:.3e} "
            f"TrainLoss={train_loss:.6f} ValLoss={val_loss:.6f} "
            f"MSE={mse_epoch:.6f} NonNeg={nonneg_epoch:.6f} Smooth={smooth_epoch:.6f} Entropy={entropy_epoch:.6f} "
            f"TrainMAE={train_mae:.6f} ValMAE={val_mae:.6f} "
            f"TrainR2={train_r2:.4f} ValR2={val_r2:.4f}"
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{save_dir}/checkpoints/best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best ValLoss={best_val:.6f}")
                break

    # ===================== Stage 3: Finetune (optional) =====================
    finetune_epochs = int(cfg["training"].get("finetune_epochs", 0))
    if finetune_epochs > 0:
        logger.info(f"=== Stage 3: Joint finetuning for {finetune_epochs} epochs (unfreeze gate + selected experts) ===")

        best_ckpt = f"{save_dir}/checkpoints/best_model.pt"
        if os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt, map_location=device))

        _set_requires_grad(gate_params, True)
        _set_requires_grad(expert_params, False)

        ft_lr_gate = float(cfg["training"].get("finetune_lr_gate", base_lr * 0.5))
        ft_lr_expert = float(cfg["training"].get("finetune_lr_expert", base_lr * 0.2))

        ft_unfreeze = cfg["training"].get("finetune_unfreeze", ["liq"])
        ft_unfreeze = [str(x).lower() for x in (ft_unfreeze or [])]

        def _match(key: str) -> bool:
            if "all" in ft_unfreeze:
                return True
            for it in ft_unfreeze:
                if it in key:
                    return True
            return False

        expert_param_groups = []
        if hasattr(model, "expert_gas") and _match("gas"):
            _set_requires_grad(model.expert_gas.parameters(), True)
            expert_param_groups += list(model.expert_gas.parameters())
        if hasattr(model, "expert_liq") and _match("liq"):
            _set_requires_grad(model.expert_liq.parameters(), True)
            expert_param_groups += list(model.expert_liq.parameters())
        if hasattr(model, "expert_crit") and _match("crit"):
            _set_requires_grad(model.expert_crit.parameters(), True)
            expert_param_groups += list(model.expert_crit.parameters())
        if hasattr(model, "expert_extra") and _match("extra"):
            _set_requires_grad(model.expert_extra.parameters(), True)
            expert_param_groups += list(model.expert_extra.parameters())

        params_gate = _filter_trainable(gate_params)
        params_expert = _filter_trainable(expert_param_groups)

        if len(params_gate) == 0 and len(params_expert) == 0:
            logger.info("[Finetune] No trainable params found. Skip finetune stage.")
        else:
            ft_opt_kind = str(cfg["training"].get("finetune_optimizer", "adamw")).lower()
            ft_weight_decay = float(cfg["training"].get("finetune_weight_decay", 5e-5))
            ft_betas = cfg["training"].get("finetune_betas", [0.9, 0.999])
            ft_betas = (float(ft_betas[0]), float(ft_betas[1])) if isinstance(ft_betas, (list, tuple)) and len(ft_betas) >= 2 else (0.9, 0.999)

            # 分组优化器：gate 和 expert 用不同 lr，但共享同一 optimizer 实例（正常、推荐）
            if ft_opt_kind == "adamw":
                optim_joint = torch.optim.AdamW(
                    [
                        {"params": params_gate, "lr": ft_lr_gate, "weight_decay": ft_weight_decay},
                        {"params": params_expert, "lr": ft_lr_expert, "weight_decay": ft_weight_decay},
                    ],
                    betas=ft_betas,
                )
            elif ft_opt_kind == "adam":
                optim_joint = torch.optim.Adam(
                    [
                        {"params": params_gate, "lr": ft_lr_gate},
                        {"params": params_expert, "lr": ft_lr_expert},
                    ],
                    betas=ft_betas,
                    weight_decay=ft_weight_decay,
                )
            else:
                optim_joint = _make_optimizer(
                    ft_opt_kind,
                    [{"params": params_gate, "lr": ft_lr_gate}, {"params": params_expert, "lr": ft_lr_expert}],
                    lr=ft_lr_gate,
                    betas=ft_betas,
                    weight_decay=ft_weight_decay,
                )

            sched_ft, sched_ft_interval = _build_lr_scheduler(cfg, optim_joint, "finetune", finetune_epochs, steps_per_epoch)

            for ft_ep in range(1, finetune_epochs + 1):
                tau = _temperature_schedule(cfg, "finetune", ft_ep, finetune_epochs)

                model.train()
                running_train_loss = 0.0
                mse_epoch = 0.0
                nonneg_epoch = 0.0
                smooth_epoch = 0.0
                entropy_epoch = 0.0
                n_batches = 0
                w_accum = None
                train_y_true, train_y_pred = [], []

                for batch in train_loader:
                    expert_ids = None
                    if len(batch) == 3:
                        x, y, expert_ids = batch
                    elif len(batch) == 2:
                        x, y = batch
                    else:
                        raise RuntimeError("Unexpected batch format in finetune stage.")
                    x = x.to(device)
                    y = y.to(device)
                    if expert_ids is not None:
                        expert_ids = expert_ids.to(device).long()

                    optim_joint.zero_grad()

                    out = model(x)
                    if isinstance(out, (list, tuple)) and len(out) >= 3:
                        w_raw, expert_outputs = out[1], out[2]
                    elif isinstance(out, dict):
                        w_raw = out.get("w_raw", None) or out.get("gate_w", None)
                        expert_outputs = out.get("expert_outputs", None)
                        if w_raw is None or expert_outputs is None:
                            raise RuntimeError("Model output dict missing gate weights or expert outputs.")
                    else:
                        raise RuntimeError("Unexpected model forward output format.")

                    w_tau = _apply_temperature_to_weights(w_raw, tau)
                    preds = (w_tau * expert_outputs).sum(dim=1, keepdim=True)

                    loss, loss_dict = criterion(preds, y, expert_id=expert_ids, gate_w=w_tau, extra=None)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optim_joint.step()

                    _step_scheduler(sched_ft, sched_ft_interval, when="batch")

                    running_train_loss += loss.item()
                    mse_epoch += loss_dict.get("MSE", 0.0)
                    nonneg_epoch += loss_dict.get("NonNeg", 0.0)
                    smooth_epoch += loss_dict.get("Smooth", 0.0)
                    entropy_epoch += loss_dict.get("Entropy", 0.0)
                    n_batches += 1

                    train_y_true.append(y.detach().cpu().numpy())
                    train_y_pred.append(preds.detach().cpu().numpy())

                    w_mean_batch = w_tau.mean(dim=0)
                    w_accum = w_mean_batch.detach() if w_accum is None else (w_accum + w_mean_batch.detach())

                train_loss = running_train_loss / max(n_batches, 1)
                mse_epoch /= max(n_batches, 1)
                nonneg_epoch /= max(n_batches, 1)
                smooth_epoch /= max(n_batches, 1)
                entropy_epoch /= max(n_batches, 1)

                train_y_true_np = np.concatenate(train_y_true, axis=0).reshape(-1)
                train_y_pred_np = np.concatenate(train_y_pred, axis=0).reshape(-1)
                train_mae = mean_absolute_error(train_y_true_np, train_y_pred_np)
                train_mse_metric = mean_squared_error(train_y_true_np, train_y_pred_np)
                train_r2 = r2_score(train_y_true_np, train_y_pred_np)

                model.eval()
                running_val_loss = 0.0
                n_val_batches = 0
                val_y_true, val_y_pred = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        expert_ids = None
                        if len(batch) == 3:
                            x, y, expert_ids = batch
                        elif len(batch) == 2:
                            x, y = batch
                        else:
                            raise RuntimeError("Unexpected batch format in finetune validation.")
                        x = x.to(device)
                        y = y.to(device)
                        if expert_ids is not None:
                            expert_ids = expert_ids.to(device).long()

                        out = model(x)
                        if isinstance(out, (list, tuple)) and len(out) >= 3:
                            w_raw, expert_outputs = out[1], out[2]
                        elif isinstance(out, dict):
                            w_raw = out.get("w_raw", None) or out.get("gate_w", None)
                            expert_outputs = out.get("expert_outputs", None)
                            if w_raw is None or expert_outputs is None:
                                raise RuntimeError("Model output dict missing gate weights or expert outputs.")
                        else:
                            raise RuntimeError("Unexpected model forward output format.")

                        w_tau = _apply_temperature_to_weights(w_raw, tau)
                        preds = (w_tau * expert_outputs).sum(dim=1, keepdim=True)

                        vloss, _ = criterion(preds, y, expert_id=expert_ids, gate_w=w_tau, extra=None)
                        running_val_loss += vloss.item()
                        n_val_batches += 1

                        val_y_true.append(y.detach().cpu().numpy())
                        val_y_pred.append(preds.detach().cpu().numpy())

                val_loss = running_val_loss / max(n_val_batches, 1)

                val_y_true_np = np.concatenate(val_y_true, axis=0).reshape(-1)
                val_y_pred_np = np.concatenate(val_y_pred, axis=0).reshape(-1)
                val_mae = mean_absolute_error(val_y_true_np, val_y_pred_np)
                val_mse_metric = mean_squared_error(val_y_true_np, val_y_pred_np)
                val_r2 = r2_score(val_y_true_np, val_y_pred_np)

                _step_scheduler(sched_ft, sched_ft_interval, when="epoch_end", metric=val_loss)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                mse_list.append(mse_epoch)
                nonneg_list.append(nonneg_epoch)
                smooth_list.append(smooth_epoch)
                entropy_list.append(entropy_epoch)
                train_mae_list.append(train_mae)
                val_mae_list.append(val_mae)
                train_mse_metric_list.append(train_mse_metric)
                val_mse_metric_list.append(val_mse_metric)
                train_r2_list.append(train_r2)
                val_r2_list.append(val_r2)

                if w_accum is not None:
                    w_mean = (w_accum / max(n_batches, 1)).cpu().numpy()
                    gate_e1_list.append(float(w_mean[0]))
                    gate_e2_list.append(float(w_mean[1]))
                    gate_e3_list.append(float(w_mean[2]))
                    gate_e4_list.append(float(w_mean[3]))
                else:
                    gate_e1_list.append(0.0)
                    gate_e2_list.append(0.0)
                    gate_e3_list.append(0.0)
                    gate_e4_list.append(0.0)

                logger.info(
                    f"[Finetune {ft_ep}/{finetune_epochs}] tau={tau:.3f} "
                    f"LR={optim_joint.param_groups[0]['lr']:.3e} "
                    f"TrainLoss={train_loss:.6f} ValLoss={val_loss:.6f} "
                    f"MSE={mse_epoch:.6f} NonNeg={nonneg_epoch:.6f} Smooth={smooth_epoch:.6f} Entropy={entropy_epoch:.6f} "
                    f"TrainMAE={train_mae:.6f} ValMAE={val_mae:.6f} "
                    f"TrainR2={train_r2:.4f} ValR2={val_r2:.4f}"
                )

    # ===================== Artifacts / plots =====================
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=train_losses, name="Train Loss"))
    fig1.add_trace(go.Scatter(y=val_losses, name="Val Loss"))
    fig1.update_layout(title="Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")
    fig1.write_html(f"{save_dir}/plots/loss_curve.html")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=mse_list, name="MSE (data term)"))
    fig2.add_trace(go.Scatter(y=nonneg_list, name="NonNeg"))
    fig2.add_trace(go.Scatter(y=smooth_list, name="Smooth"))
    fig2.add_trace(go.Scatter(y=entropy_list, name="Entropy"))
    fig2.update_layout(title="Loss Components", xaxis_title="Epoch", yaxis_title="Value")
    fig2.write_html(f"{save_dir}/plots/loss_components.html")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=gate_e1_list, name="Expert1"))
    fig3.add_trace(go.Scatter(y=gate_e2_list, name="Expert2"))
    fig3.add_trace(go.Scatter(y=gate_e3_list, name="Expert3"))
    fig3.add_trace(go.Scatter(y=gate_e4_list, name="Expert4"))
    fig3.update_layout(title="Gate Weights (Mean, temperature-adjusted)", xaxis_title="Epoch", yaxis_title="Weight")
    fig3.write_html(f"{save_dir}/plots/gate_weights.html")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=train_mae_list, name="Train MAE"))
    fig4.add_trace(go.Scatter(y=val_mae_list, name="Val MAE"))
    fig4.update_layout(title="MAE Curve", xaxis_title="Epoch", yaxis_title="MAE")
    fig4.write_html(f"{save_dir}/plots/mae_curve.html")

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(y=train_mse_metric_list, name="Train MSE"))
    fig5.add_trace(go.Scatter(y=val_mse_metric_list, name="Val MSE"))
    fig5.update_layout(title="MSE Curve", xaxis_title="Epoch", yaxis_title="MSE")
    fig5.write_html(f"{save_dir}/plots/mse_curve.html")

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(y=train_r2_list, name="Train R2"))
    fig6.add_trace(go.Scatter(y=val_r2_list, name="Val R2"))
    fig6.update_layout(title="R2 Curve", xaxis_title="Epoch", yaxis_title="R2")
    fig6.write_html(f"{save_dir}/plots/r2_curve.html")

    df = pd.DataFrame({
        "TrainLoss": train_losses,
        "ValLoss": val_losses,
        "DataLoss": mse_list,
        "NonNeg": nonneg_list,
        "Smooth": smooth_list,
        "Entropy": entropy_list,
        "Train_MAE": train_mae_list,
        "Val_MAE": val_mae_list,
        "Train_MSE_metric": train_mse_metric_list,
        "Val_MSE_metric": val_mse_metric_list,
        "Train_R2": train_r2_list,
        "Val_R2": val_r2_list,
        "Gate_Expert1": gate_e1_list,
        "Gate_Expert2": gate_e2_list,
        "Gate_Expert3": gate_e3_list,
        "Gate_Expert4": gate_e4_list,
    })
    df.to_csv(f"{save_dir}/plots/training_metrics.csv", index=False)
    logger.info(f"Training complete. Artifacts written to {save_dir}")