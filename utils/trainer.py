import os
import torch
import plotly.graph_objects as go
import pandas as pd

def train_model(model, dataloaders, criterion, optimizer, cfg, device, logger):
    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    train_losses, val_losses = [], []
    mse_list, nonneg_list, smooth_list = [], [], []
    gate_gas_list, gate_liq_list, gate_crit_list = [], [], []

    best_val = float("inf")
    patience = 0
    max_epochs = cfg["training"]["epochs"]

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_loss = tr_mse = tr_non = tr_smooth = 0.0
        all_w = []

        for X, y in dataloaders["train"]:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds, w, _ = model(X)
            loss, parts = criterion(preds, y)
            loss.backward()
            optimizer.step()

            bs = X.size(0)
            tr_loss += float(loss.detach().cpu()) * bs
            tr_mse += parts["MSE"] * bs
            tr_non += parts["NonNeg"] * bs
            tr_smooth += parts["Smooth"] * bs
            all_w.append(w.detach().cpu())

        n_train = len(dataloaders["train"].dataset)
        train_loss = tr_loss / max(n_train, 1)
        mse_epoch = tr_mse / max(n_train, 1)
        nonneg_epoch = tr_non / max(n_train, 1)
        smooth_epoch = tr_smooth / max(n_train, 1)

        if len(all_w) > 0:
            w_all = torch.cat(all_w, dim=0).numpy()
            gate_gas, gate_liq, gate_crit = w_all.mean(axis=0).tolist()
        else:
            gate_gas = gate_liq = gate_crit = 0.0

        # validation
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for X, y in dataloaders["val"]:
                X, y = X.to(device), y.to(device)
                preds, _, _ = model(X)
                vloss, _ = criterion(preds, y)
                val_sum += float(vloss.detach().cpu()) * X.size(0)
        val_loss = val_sum / max(len(dataloaders["val"].dataset), 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        mse_list.append(mse_epoch)
        nonneg_list.append(nonneg_epoch)
        smooth_list.append(smooth_epoch)
        gate_gas_list.append(gate_gas)
        gate_liq_list.append(gate_liq)
        gate_crit_list.append(gate_crit)

        logger.info(
            f"Epoch {epoch}/{max_epochs} | "
            f"Train={train_loss:.6f} Val={val_loss:.6f} | "
            f"MSE={mse_epoch:.6f} NonNeg={nonneg_epoch:.6f} Smooth={smooth_epoch:.6f} | "
            f"Gate=[{gate_gas:.3f},{gate_liq:.3f},{gate_crit:.3f}]"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{save_dir}/checkpoints/best_model.pt")
            patience = 0
        else:
            patience += 1
            if patience > cfg["training"]["early_stopping_patience"]:
                logger.info("Early stopping triggered.")
                break

    # Plots
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, name="Train Loss"))
    fig.add_trace(go.Scatter(y=val_losses, name="Val Loss"))
    fig.update_layout(title="Training & Validation Loss", xaxis_title="Epoch", yaxis_title="Loss")
    fig.write_html(f"{save_dir}/plots/training_loss.html")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=mse_list, name="MSE"))
    fig2.add_trace(go.Scatter(y=nonneg_list, name="NonNeg"))
    fig2.add_trace(go.Scatter(y=smooth_list, name="Smooth"))
    fig2.update_layout(title="Loss Component Evolution", xaxis_title="Epoch", yaxis_title="Loss")
    fig2.write_html(f"{save_dir}/plots/loss_components.html")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=gate_gas_list, name="Gas Expert"))
    fig3.add_trace(go.Scatter(y=gate_liq_list, name="Liquid Expert"))
    fig3.add_trace(go.Scatter(y=gate_crit_list, name="Critical Expert"))
    fig3.update_layout(title="Average Gating Weights Evolution", xaxis_title="Epoch",
                       yaxis_title="Average Weight", yaxis_range=[0, 1])
    fig3.write_html(f"{save_dir}/plots/gating_weights.html")

    # CSV
    df = pd.DataFrame({
        "Epoch": list(range(1, len(train_losses) + 1)),
        "TrainLoss": train_losses, "ValLoss": val_losses,
        "MSE": mse_list, "NonNeg": nonneg_list, "Smooth": smooth_list,
        "Gate_Gas": gate_gas_list, "Gate_Liquid": gate_liq_list, "Gate_Critical": gate_crit_list
    })
    df.to_csv(f"{save_dir}/plots/training_metrics.csv", index=False)
    logger.info(f"Training complete. Artifacts written to {save_dir}")