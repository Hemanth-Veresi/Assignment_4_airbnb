import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_metric_bar(results_df, metric="rmse", title="Model Comparison", save_path=None):
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=results_df.index, y=results_df[metric], palette="viridis")

    plt.title(title)
    plt.ylabel(metric.upper())
    plt.xlabel("Model")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.show()


def plot_actual_vs_pred(actual, pred, title="Actual vs Predicted", save_path=None):
    """
    Scatter plot showing how predictions line up with actual values.
    """

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=actual, y=pred, alpha=0.4)

    # Plot the perfect equal line
    max_val = max(max(actual), max(pred))
    min_val = min(min(actual), min(pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")

    plt.title(title)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.show()


def plot_residuals(actual, pred, title="Residual Plot", save_path=None):
    """
    Visualize residuals (prediction errors).
    """

    residuals = actual - pred

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="steelblue", bins=40)

    plt.title(title)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.show()


def plot_heatmap(matrix, row_labels, col_labels, title="Cross-Tier RMSE Heatmap", save_path=None):
    """
    Create a heatmap of RMSE scores.
    Useful for evaluating how models trained on one tier perform on other tiers.
    """

    plt.figure(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="mako",
                xticklabels=col_labels, yticklabels=row_labels)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.show()
