"""
Step 3: Evaluate MLP Models on Independent Test Set
Evaluate trained MLP models on the independent test set

Usage:
    # Evaluate all models
    python aging_scripts/03_evaluate_models.py

    # Evaluate selected models
    python aging_scripts/03_evaluate_models.py --model global
    python aging_scripts/03_evaluate_models.py --model per_part
"""

import os
import json
import logging
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EMBEDDING_DIR = "./embedding_results"
OUTPUT_DIR = "./output_evaluation"


# ===================== MLP model definition =====================
# (must match the training scripts)
class AgeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super(AgeMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[AgeMLP, dict]:
    """Load trained MLP model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    input_dim = checkpoint['input_dim']
    hidden_dims = checkpoint['hidden_dims']
    dropout = checkpoint['dropout']

    model = AgeMLP(input_dim, hidden_dims, dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('training_config', {})


def load_test_data():
    """Load test embeddings"""
    test_file = os.path.join(
        EMBEDDING_DIR,
        "cell_embedding_Task1_Independent.Test_GSE134355_n32000_TranscriptFormer_input.parquet"
    )
    df = pd.read_parquet(test_file)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values
    y = df["age"].values if "age" in df.columns else None
    return X, y, df


def predict(model, X, device, batch_size=512):
    """Generate predictions"""
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            pred = model(batch).cpu().numpy()
            preds.extend(pred)
    return np.array(preds)


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    pcc, p_value = pearsonr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return {"MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "PCC": round(float(pcc), 4),
            "R2": round(float(r2), 4)}


def plot_predictions(y_true, predictions_dict, save_path):
    """Plot predicted vs true age for all models"""
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, pred) in zip(axes, predictions_dict.items()):
        metrics = calculate_metrics(y_true, pred)
        ax.scatter(y_true, pred, alpha=0.3, s=5, c='steelblue')
        # Diagonal line
        min_val, max_val = y_true.min(), y_true.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
        ax.set_xlabel("True Age", fontsize=12)
        ax.set_ylabel("Predicted Age", fontsize=12)
        ax.set_title(f"{name}\nMAE={metrics['MAE']:.3f} | RMSE={metrics['RMSE']:.3f} | PCC={metrics['PCC']:.3f} | R^2={metrics['R2']:.3f}",
                     fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Scatter plot saved to {save_path}")


def plot_comparison_table(metrics_dict, save_path):
    """Plot a comparison table of all models"""
    rows = []
    for name, metrics in metrics_dict.items():
        rows.append({
            "Model": name,
            "MAE (lower)": metrics['MAE'],
            "RMSE (lower)": metrics['RMSE'],
            "PCC (higher)": metrics['PCC'],
            "R^2 (higher)": metrics['R2']
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, len(df) * 0.6 + 1))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Highlight best values
    for i in range(1, len(df) + 1):
        for j in range(1, len(df.columns)):
            pass  # styling logic if needed

    plt.title("Model Comparison on Independent Test Set", fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Comparison table saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP models on test set")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "global", "per_part"],
                        help="Which model to evaluate")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device ID")
    parser.add_argument("--save_plots", action="store_true", default=True,
                        help="Save prediction scatter plots")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load test data
    X_test, y_test, test_df = load_test_data()
    logger.info(f"Test set: {X_test.shape[0]} samples, embedding dim: {X_test.shape[1]}")
    logger.info(f"Age range: {y_test.min():.2f} - {y_test.max():.2f}")

    # Model paths
    model_paths = {
        "Global Batching": "./output_mlp_global/mlp_model_global.pt",
        "Per-Part Batching": "./output_mlp_per_part/mlp_model_per_part.pt",
    }

    if args.model != "all":
        model_paths = {k: v for k, v in model_paths.items() if args.model in k.lower()}

    # Evaluate each model
    all_metrics = {}
    all_predictions = {}
    output_preds = {}

    for name, path in model_paths.items():
        if not os.path.exists(path):
            logger.warning(f"Model not found: {path}  -- skipping")
            continue

        logger.info(f"\n--- Evaluating: {name} ---")
        model, train_cfg = load_model(path, device)
        logger.info(f"  Training config: {train_cfg}")

        preds = predict(model, X_test, device)
        metrics = calculate_metrics(y_test, preds)

        logger.info(f"  MAE:  {metrics['MAE']:.4f}")
        logger.info(f"  RMSE: {metrics['RMSE']:.4f}")
        logger.info(f"  PCC:  {metrics['PCC']:.4f}")
        logger.info(f"  R^2:   {metrics['R2']:.4f}")

        all_metrics[name] = metrics
        all_predictions[name] = preds

        # Save predictions
        safe_name = name.lower().replace(" ", "_")
        pred_df = pd.DataFrame({
            "true_age": y_test,
            "predicted_age": preds,
        })
        pred_df.to_csv(f"{OUTPUT_DIR}/predictions_{safe_name}.csv", index=False)

        # Save metrics
        with open(f"{OUTPUT_DIR}/metrics_{safe_name}.json", 'w') as f:
            json.dump({**metrics, **train_cfg}, f, indent=2)

    # Summary comparison
    logger.info("\n" + "=" * 60)
    logger.info("Model Comparison Summary")
    logger.info("=" * 60)
    logger.info(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'PCC':>8} {'R^2':>8}")
    logger.info("-" * 54)
    for name, m in all_metrics.items():
        logger.info(f"{name:<20} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['PCC']:>8.4f} {m['R2']:>8.4f}")
    logger.info("=" * 60)

    # Save summary comparison JSON
    summary_file = f"{OUTPUT_DIR}/evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Generate plots
    if args.save_plots and all_predictions:
        plot_predictions(y_test, all_predictions,
                        f"{OUTPUT_DIR}/scatter_comparison.png")
        plot_comparison_table(all_metrics,
                              f"{OUTPUT_DIR}/metrics_table.png")

    logger.info(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()