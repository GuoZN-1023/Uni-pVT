import os
import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

def visualize_phase_distribution(model, dataset, save_path, method="tsne", device="cpu"):
    model.eval()
    X = dataset.X.to(device)
    with torch.no_grad():
        _, w, _ = model(X)
    w = w.cpu().numpy()

    X_np = dataset.X.cpu().numpy()
    if method.lower() == "umap" and HAS_UMAP:
        reducer = umap.UMAP(random_state=42)
        emb = reducer.fit_transform(X_np)
        name = "UMAP"
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        emb = reducer.fit_transform(X_np)
        name = "t-SNE"

    df = pd.DataFrame({"x": emb[:,0], "y": emb[:,1], "Gas": w[:,0], "Liquid": w[:,1], "Critical": w[:,2]})
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for col in ["Gas", "Liquid", "Critical"]:
        fig = px.scatter(df, x="x", y="y", color=col, color_continuous_scale="Viridis",
                         title=f"{name} Projection - {col} Expert Weight", width=800, height=650)
        fig.write_html(save_path.replace(".html", f"_{col}.html"))