# utils/trainer.py
import os
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
    We approximate logits by log(w) (up to constant), giving:
      softmax(log(w)/tau) == normalize(w^(1/tau))
    """
    tau = float(tau)
    if tau <= 0:
        tau = 1.0
    w = torch.clamp(w, 1e-12, 1.0)
    p = 1.0 / tau
    w2 = w ** p
    w2 = w2 / (w2.sum(dim=1, keepdim=True) + 1e-12)
    return w2


def _temperature_schedule(cfg: dict, stage: str, epoch_idx: int, total_epochs: int) -> float:
    """
    stage: "stage2" or "finetune"
    config example:
      training:
        gate_temp_start: 1.0
        gate_temp_end: 0.5
        gate_temp_mode: "linear"
    """
    tr = cfg.get("training", {})
    t0 = float(tr.get("gate_temp_start", 1.0))
    t1 = float(tr.get("gate_temp_end", 1.0))
    mode = str(tr.get("gate_temp_mode", "linear")).lower()

    if total_epochs <= 1:
        return t1

    alpha = float(epoch_idx - 1) / float(total_epochs - 1)
    if mode in ("cos", "cosine"):
        # cosine anneal from t0 -> t1
        alpha = 0.5 * (1 - np.cos(np.pi * alpha))
    # linear default
    return t0 + (t1 - t0) * alpha


def train_model(model, dataloaders, criterion, optimizer_unused, cfg, device, logger):
    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    total_epochs = int(cfg["training"]["epochs"])
    pretrain_epochs = int(cfg["training"].get("pretrain_epochs", 0))
    patience = int(cfg["training"]["early_stopping_patience"])

    # expert / gate params
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

    lr = float(cfg["training"]["learning_rate"])
    optim_experts = torch.optim.Adam(expert_params, lr=lr) if pretrain_epochs > 0 else None
    optim_gate = torch.optim.Adam(gate_params, lr=lr)

    # records
    train_losses, val_losses = [], []
    mse_list, nonneg_list, smooth_list, entropy_list = [], [], [], []
    gate_e1_list, gate_e2_list, gate_e3_list, gate_e4_list = [], [], [], []
    train_mae_list, val_mae_list = [], []
    train_mse_metric_list, val_mse_metric_list = [], []
    train_r2_list, val_r2_list = [], []

    # ========= Stage 1: pretrain experts by hard routing =========
    if pretrain_epochs > 0:
        logger.info(f"=== Stage 1: Pretraining experts for {pretrain_epochs} epochs (hard routing by 'no') ===")
        for p in gate_params:
            p.requires_grad = False

        for epoch in range(1, pretrain_epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                if len(batch) == 3:
                    x, y, expert_ids = batch
                else:
                    raise RuntimeError("Pretraining requires (x, y, expert_id) in dataset.")
                x = x.to(device)
                y = y.to(device)
                expert_ids = expert_ids.to(device).long()  # 1..4

                optim_experts.zero_grad()
                _, _, expert_outputs = model(x)  # [B,4]
                idx = expert_ids - 1
                selected = expert_outputs.gather(1, idx.view(-1, 1))  # [B,1]
                loss, _ = criterion(selected, y, expert_id=expert_ids, gate_w=None, extra=None)
                loss.backward()
                optim_experts.step()

                running_loss += loss.item()
                n_batches += 1

            avg_loss = running_loss / max(n_batches, 1)
            logger.info(f"[Pretrain] Epoch {epoch}/{pretrain_epochs}  TrainLoss={avg_loss:.6f}")

        # freeze experts, train gate only
        for p in expert_params:
            p.requires_grad = False
        for p in gate_params:
            p.requires_grad = True

    # ========= Stage 2: train gate (and fused output) with early stopping =========
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

            preds_raw, w_raw, expert_outputs = model(x)  # preds_raw may be unused
            w_tau = _apply_temperature_to_weights(w_raw, tau)
            preds = (w_tau * expert_outputs).sum(dim=1, keepdim=True)  # (B,1)

            loss, loss_dict = criterion(preds, y, expert_id=expert_ids, gate_w=w_tau, extra=None)
            loss.backward()
            optim_gate.step()

            running_train_loss += loss.item()
            mse_epoch += loss_dict.get("MSE", 0.0)
            nonneg_epoch += loss_dict.get("NonNeg", 0.0)
            smooth_epoch += loss_dict.get("Smooth", 0.0)
            entropy_epoch += loss_dict.get("Entropy", 0.0)
            n_batches += 1

            train_y_true.append(y.detach().cpu().numpy())
            train_y_pred.append(preds.detach().cpu().numpy())

            w_mean_batch = w_tau.mean(dim=0)  # use temperature-adjusted weights
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

        # ---------- val ----------
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

                preds_raw, w_raw, expert_outputs = model(x)
                w_tau = _apply_temperature_to_weights(w_raw, tau)
                preds = (w_tau * expert_outputs).sum(dim=1, keepdim=True)

                loss, _ = criterion(preds, y, expert_id=expert_ids, gate_w=w_tau, extra=None)
                running_val_loss += loss.item()
                n_val_batches += 1
                val_y_true.append(y.detach().cpu().numpy())
                val_y_pred.append(preds.detach().cpu().numpy())

        val_loss = running_val_loss / max(n_val_batches, 1)
        val_y_true_np = np.concatenate(val_y_true, axis=0).reshape(-1)
        val_y_pred_np = np.concatenate(val_y_pred, axis=0).reshape(-1)
        val_mae = mean_absolute_error(val_y_true_np, val_y_pred_np)
        val_mse_metric = mean_squared_error(val_y_true_np, val_y_pred_np)
        val_r2 = r2_score(val_y_true_np, val_y_pred_np)

        # record gate weights mean
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
            f"TrainLoss={train_loss:.6f} ValLoss={val_loss:.6f} "
            f"MSE={mse_epoch:.6f} NonNeg={nonneg_epoch:.6f} Smooth={smooth_epoch:.6f} Entropy={entropy_epoch:.6f} "
            f"TrainMAE={train_mae:.6f} ValMAE={val_mae:.6f} "
            f"TrainR2={train_r2:.4f} ValR2={val_r2:.4f}"
        )

        # early stopping by val_loss
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{save_dir}/checkpoints/best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best val loss={best_val:.6f}")
                break

    # ========= Stage 3: joint finetuning (unfreeze gate + selected experts) =========
    finetune_epochs = int(cfg["training"].get("finetune_epochs", 0))
    if finetune_epochs > 0:
        logger.info(f"=== Stage 3: Joint finetuning for {finetune_epochs} epochs (unfreeze gate + selected experts) ===")

        best_ckpt = f"{save_dir}/checkpoints/best_model.pt"
        if os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt, map_location=device))

        ft_lr_gate = float(cfg["training"].get("finetune_lr_gate", lr * 0.5))
        ft_lr_expert = float(cfg["training"].get("finetune_lr_expert", lr * 0.2))
        ft_unfreeze = cfg["training"].get("finetune_unfreeze", ["liq", "extra"])
        if isinstance(ft_unfreeze, str):
            ft_unfreeze = [ft_unfreeze]
        ft_unfreeze = [str(s).lower() for s in ft_unfreeze]

        # freeze all then unfreeze
        _set_requires_grad(expert_params, False)
        _set_requires_grad(gate_params, False)
        _set_requires_grad(gate_params, True)

        expert_param_groups = []

        def _match(name: str) -> bool:
            n = name.lower()
            return ("all" in ft_unfreeze) or any(u in n for u in ft_unfreeze)

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
        params_exp = _filter_trainable(expert_param_groups)

        joint_groups = []
        if params_gate:
            joint_groups.append({"params": params_gate, "lr": ft_lr_gate})
        if params_exp:
            joint_groups.append({"params": params_exp, "lr": ft_lr_expert})

        if not joint_groups:
            logger.info("[Finetune] No trainable params found. Skip finetune stage.")
        else:
            optim_joint = torch.optim.Adam(joint_groups)

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

                    preds_raw, w_raw, expert_outputs = model(x)
                    w_tau = _apply_temperature_to_weights(w_raw, tau)
                    preds = (w_tau * expert_outputs).sum(dim=1, keepdim=True)

                    loss, loss_dict = criterion(preds, y, expert_id=expert_ids, gate_w=w_tau, extra=None)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optim_joint.step()

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

                # val
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

                        preds_raw, w_raw, expert_outputs = model(x)
                        w_tau = _apply_temperature_to_weights(w_raw, tau)
                        preds = (w_tau * expert_outputs).sum(dim=1, keepdim=True)

                        loss, _ = criterion(preds, y, expert_id=expert_ids, gate_w=w_tau, extra=None)
                        running_val_loss += loss.item()
                        n_val_batches += 1
                        val_y_true.append(y.detach().cpu().numpy())
                        val_y_pred.append(preds.detach().cpu().numpy())

                val_loss = running_val_loss / max(n_val_batches, 1)
                val_y_true_np = np.concatenate(val_y_true, axis=0).reshape(-1)
                val_y_pred_np = np.concatenate(val_y_pred, axis=0).reshape(-1)
                val_mae = mean_absolute_error(val_y_true_np, val_y_pred_np)
                val_mse_metric = mean_squared_error(val_y_true_np, val_y_pred_np)
                val_r2 = r2_score(val_y_true_np, val_y_pred_np)

                # record gate weights mean
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
                    f"[Finetune {ft_ep}/{finetune_epochs}] tau={tau:.3f} "
                    f"TrainLoss={train_loss:.6f} ValLoss={val_loss:.6f} "
                    f"MSE={mse_epoch:.6f} NonNeg={nonneg_epoch:.6f} Smooth={smooth_epoch:.6f} Entropy={entropy_epoch:.6f} "
                    f"TrainMAE={train_mae:.6f} ValMAE={val_mae:.6f} "
                    f"TrainR2={train_r2:.4f} ValR2={val_r2:.4f}"
                )

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    torch.save(model.state_dict(), best_ckpt)

            if os.path.exists(best_ckpt):
                model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # ========= write plots & csv =========
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=train_losses, name="Train Loss"))
    fig1.add_trace(go.Scatter(y=val_losses, name="Val Loss"))
    fig1.update_layout(title="Train / Val Loss", xaxis_title="Epoch", yaxis_title="Loss")
    fig1.write_html(f"{save_dir}/plots/loss_curve.html")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=mse_list, name="DataLoss (weighted)"))
    fig2.add_trace(go.Scatter(y=nonneg_list, name="NonNeg Penalty"))
    fig2.add_trace(go.Scatter(y=smooth_list, name="Smooth Penalty"))
    fig2.add_trace(go.Scatter(y=entropy_list, name="Entropy Penalty"))
    fig2.update_layout(title="Loss Components", xaxis_title="Epoch", yaxis_title="Value")
    fig2.write_html(f"{save_dir}/plots/loss_components.html")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=gate_e1_list, name="Gate Expert1"))
    fig3.add_trace(go.Scatter(y=gate_e2_list, name="Gate Expert2"))
    fig3.add_trace(go.Scatter(y=gate_e3_list, name="Gate Expert3"))
    fig3.add_trace(go.Scatter(y=gate_e4_list, name="Gate Expert4"))
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
