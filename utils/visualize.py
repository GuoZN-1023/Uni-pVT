import os
import pandas as pd
import plotly.graph_objects as go

def plot_predictions(y_true, y_pred, save_html_path: str):
    os.makedirs(os.path.dirname(save_html_path), exist_ok=True)
    df = pd.DataFrame({"True Z": y_true, "Predicted Z": y_pred})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="Predictions"))
    fig.add_trace(go.Scatter(x=y_true, y=y_true, mode="lines", name="y=x", line=dict(dash="dash")))
    fig.update_layout(title="Predicted vs True Z", xaxis_title="True Z", yaxis_title="Predicted Z")
    fig.write_html(save_html_path)
    df.to_csv(save_html_path.replace(".html", ".csv"), index=False)