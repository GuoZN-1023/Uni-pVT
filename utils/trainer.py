# utils/trainer.py
import os
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

    # 拆分专家 / gate 参数
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

    # 各种记录
    train_losses, val_losses = [], []
    mse_list, nonneg_list, smooth_list = [], [], []
    calib_bias_list, calib_slope_list = [], []
    gate_e1_list, gate_e2_list, gate_e3_list, gate_e4_list = [], [], [], []

    train_mae_list, val_mae_list = [], []
    train_mse_metric_list, val_mse_metric_list = [], []
    train_r2_list, val_r2_list = [], []

    # ========= 阶段一：按 no 硬分区预训练专家 =========
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
                _, _, expert_outputs = model(x)           # [B,4]
                idx = expert_ids - 1                      # 转成 0..3
                selected = expert_outputs.gather(1, idx.view(-1, 1))  # [B,1]
                loss, _ = criterion(selected, y)
                loss.backward()
                optim_experts.step()

                running_loss += loss.item()
                n_batches += 1

            avg_loss = running_loss / max(n_batches, 1)
            logger.info(f"[Pretrain] Epoch {epoch}/{pretrain_epochs}  TrainLoss={avg_loss:.6f}")

        # 冻结专家，只训练 gate
        for p in expert_params:
            p.requires_grad = False
        for p in gate_params:
            p.requires_grad = True

    # ========= 阶段二：冻结专家，仅训练 gate + MoE =========
    logger.info(f"=== Stage 2: Training gate + MoE for up to {total_epochs} epochs with early stopping ===")
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, total_epochs + 1):
        model.train()
        running_train_loss = 0.0
        mse_epoch = 0.0
        nonneg_epoch = 0.0
        smooth_epoch = 0.0
        calib_bias_epoch = 0.0
        calib_slope_epoch = 0.0
        n_batches = 0
        w_accum = None

        train_y_true, train_y_pred = [], []

        # ---------- 训练 ----------
        for batch in train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            elif len(batch) == 2:
                x, y = batch
            else:
                raise RuntimeError("Unexpected batch format in training stage.")

            x = x.to(device)
            y = y.to(device)

            optim_gate.zero_grad()
            preds, w, _ = model(x)
            loss, loss_dict = criterion(preds, y)
            loss.backward()
            optim_gate.step()

            running_train_loss += loss.item()
            mse_epoch += loss_dict["MSE"]
            nonneg_epoch += loss_dict["NonNeg"]
            smooth_epoch += loss_dict["Smooth"]
            calib_bias_epoch += loss_dict.get("CalibBias", 0.0)
            calib_slope_epoch += loss_dict.get("CalibSlope", 0.0)
            n_batches += 1

            train_y_true.append(y.detach().cpu().numpy())
            train_y_pred.append(preds.detach().cpu().numpy())

            w_mean_batch = w.mean(dim=0)  # [4]
            if w_accum is None:
                w_accum = w_mean_batch.detach()
            else:
                w_accum += w_mean_batch.detach()

        train_loss = running_train_loss / max(n_batches, 1)
        mse_epoch /= max(n_batches, 1)
        nonneg_epoch /= max(n_batches, 1)
        smooth_epoch /= max(n_batches, 1)
        calib_bias_epoch /= max(n_batches, 1)
        calib_slope_epoch /= max(n_batches, 1)

        # 训练集 MAE/MSE/R2
        if len(train_y_true) > 0:
            y_true_tr = np.concatenate(train_y_true, axis=0).reshape(-1)
            y_pred_tr = np.concatenate(train_y_pred, axis=0).reshape(-1)
            train_mae = mean_absolute_error(y_true_tr, y_pred_tr)
            train_mse_metric = mean_squared_error(y_true_tr, y_pred_tr)
            train_r2 = r2_score(y_true_tr, y_pred_tr)
        else:
            train_mae = train_mse_metric = train_r2 = float("nan")

        # ---------- 验证 ----------
        model.eval()
        val_running = 0.0
        n_val_batches = 0
        val_y_true, val_y_pred = [], []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x_val, y_val, _ = batch
                elif len(batch) == 2:
                    x_val, y_val = batch
                else:
                    raise RuntimeError("Unexpected batch format in validation stage.")

                x_val = x_val.to(device)
                y_val = y_val.to(device)
                preds_val, _, _ = model(x_val)
                val_loss, _ = criterion(preds_val, y_val)
                val_running += val_loss.item()
                n_val_batches += 1

                val_y_true.append(y_val.detach().cpu().numpy())
                val_y_pred.append(preds_val.detach().cpu().numpy())

        val_loss = val_running / max(n_val_batches, 1)

        if len(val_y_true) > 0:
            y_true_val = np.concatenate(val_y_true, axis=0).reshape(-1)
            y_pred_val = np.concatenate(val_y_pred, axis=0).reshape(-1)
            val_mae = mean_absolute_error(y_true_val, y_pred_val)
            val_mse_metric = mean_squared_error(y_true_val, y_pred_val)
            val_r2 = r2_score(y_true_val, y_pred_val)
        else:
            val_mae = val_mse_metric = val_r2 = float("nan")

        # gate 权重平均
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

        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        mse_list.append(mse_epoch)
        nonneg_list.append(nonneg_epoch)
        smooth_list.append(smooth_epoch)
        calib_bias_list.append(calib_bias_epoch)
        calib_slope_list.append(calib_slope_epoch)

        train_mae_list.append(train_mae)
        val_mae_list.append(val_mae)
        train_mse_metric_list.append(train_mse_metric)
        val_mse_metric_list.append(val_mse_metric)
        train_r2_list.append(train_r2)
        val_r2_list.append(val_r2)

        logger.info(
            f"[Epoch {epoch}/{total_epochs}] "
            f"TrainLoss={train_loss:.6f} ValLoss={val_loss:.6f} "
            f"MSE={mse_epoch:.6f} NonNeg={nonneg_epoch:.6f} Smooth={smooth_epoch:.6f} "
            f"CalibBias={calib_bias_epoch:.6f} CalibSlope={calib_slope_epoch:.6f} "
            f"TrainMAE={train_mae:.6f} ValMAE={val_mae:.6f} "
            f"TrainR2={train_r2:.4f} ValR2={val_r2:.4f} "
            f"GateW={[gate_e1_list[-1], gate_e2_list[-1], gate_e3_list[-1], gate_e4_list[-1]]}"
        )

        # early stopping（按 ValLoss）
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{save_dir}/checkpoints/best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. Best val loss={best_val:.6f}"
                )
                break

    # ========= 训练结束：画图 & 写 CSV =========
    # loss 曲线
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=train_losses, name="Train Loss"))
    fig1.add_trace(go.Scatter(y=val_losses, name="Val Loss"))
    fig1.update_layout(title="Train / Val Loss", xaxis_title="Epoch", yaxis_title="Loss")
    fig1.write_html(f"{save_dir}/plots/loss_curve.html")

    # loss 各分量
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=mse_list, name="MSE (data term)"))
    fig2.add_trace(go.Scatter(y=nonneg_list, name="NonNeg Penalty"))
    fig2.add_trace(go.Scatter(y=smooth_list, name="Smoothness Penalty"))
    fig2.add_trace(go.Scatter(y=calib_bias_list, name="Calib Bias Penalty"))
    fig2.add_trace(go.Scatter(y=calib_slope_list, name="Calib Slope Penalty"))
    fig2.update_layout(title="Loss Components", xaxis_title="Epoch", yaxis_title="Value")
    fig2.write_html(f"{save_dir}/plots/loss_components.html")

    # gate 权重演化
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=gate_e1_list, name="Expert 1 (Gas)"))
    fig3.add_trace(go.Scatter(y=gate_e2_list, name="Expert 2 (Liquid)"))
    fig3.add_trace(go.Scatter(y=gate_e3_list, name="Expert 3 (Critical)"))
    fig3.add_trace(go.Scatter(y=gate_e4_list, name="Expert 4 (Extra)"))
    fig3.update_layout(
        title="Average Gating Weights Evolution",
        xaxis_title="Epoch",
        yaxis_title="Average Weight",
        yaxis_range=[0, 1],
    )
    fig3.write_html(f"{save_dir}/plots/gating_weights.html")

    # MAE / R² 曲线
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=train_mae_list, name="Train MAE"))
    fig4.add_trace(go.Scatter(y=val_mae_list, name="Val MAE"))
    fig4.add_trace(go.Scatter(y=train_r2_list, name="Train R²"))
    fig4.add_trace(go.Scatter(y=val_r2_list, name="Val R²"))
    fig4.update_layout(
        title="Regression Metrics over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Metric Value",
    )
    fig4.write_html(f"{save_dir}/plots/metrics_curve.html")

    # 写 CSV，每个 epoch 一行
    df = pd.DataFrame({
        "Epoch": list(range(1, len(train_losses) + 1)),
        "TrainLoss": train_losses,
        "ValLoss": val_losses,
        "MSE_component": mse_list,
        "NonNeg": nonneg_list,
        "Smooth": smooth_list,
        "CalibBias": calib_bias_list,
        "CalibSlope": calib_slope_list,
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
