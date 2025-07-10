import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import wandb

def evaluate_predictions_combined(test_preds: np.ndarray, test_true: np.ndarray, dataset_name: str = "Test", wandb_name: str = None,  wandb_project: str = None, output_path: str = None) -> dict:
    """
    Evaluate model predictions and generate a combined plot image with:
      - First row: True vs. Predicted scatter plots for each target.
      - Second row: Residual plots for each target.
    Also computes RMSE, MAE, and R² metrics for each target.

    Parameters
    ----------
    test_preds : np.ndarray
        Model predictions of shape (N, n_targets).
    test_true : np.ndarray
        True target values of shape (N, n_targets).
    dataset_name : str
        Name of dataset (e.g., "Validation" or "Test").

    Returns
    -------
    metrics_dict : dict
        A dictionary with keys 'rmse', 'mae', and 'r2'. Each maps to a list of metrics for each target.
    """
    # Fetch W&B run name (fallback to "unnamed_experiment" if not running inside a W&B run)
    wandb_name = wandb_name if wandb_name else "unnamed_experiment"

    # Construct output filename
    output_filename = f"{wandb_name}_{dataset_name}.png"
    
    n_targets = test_preds.shape[1]
    rmse_list, mae_list, r2_list = [], [], []
    
    # Create a figure with 2 rows and n_targets columns.
    fig, axes = plt.subplots(nrows=2, ncols=n_targets, figsize=(5 * n_targets, 10))
    # Ensure axes is a 2D array even when n_targets==1.
    if n_targets == 1:
        axes = np.array(axes).reshape(2, 1)
    
    for i in range(n_targets):
        true_vals = test_true[:, i]
        preds_vals = test_preds[:, i]
        
        # Compute evaluation metrics.
        rmse = np.sqrt(mean_squared_error(true_vals, preds_vals))
        mae = mean_absolute_error(true_vals, preds_vals)
        r2 = r2_score(true_vals, preds_vals)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        
        # Plot True vs. Predicted in the first row.
        ax1 = axes[0, i]
        ax1.scatter(true_vals, preds_vals, alpha=0.7)
        # Plot the ideal line y = x.
        x_min = min(true_vals.min(), preds_vals.min())
        x_max = max(true_vals.max(), preds_vals.max())
        ax1.plot([x_min, x_max], [x_min, x_max], "r--", label="Ideal")
        ax1.set_xlabel("True Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title(f"{dataset_name} - Target {i+1}\nRMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")
        ax1.legend()
        
        # Plot Residuals in the second row.
        ax2 = axes[1, i]
        residuals = true_vals - preds_vals
        ax2.scatter(true_vals, residuals, alpha=0.7)
        ax2.axhline(0, color="r", linestyle="--")
        ax2.set_xlabel("True Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title(f"{dataset_name} - Residuals for Target {i+1}")
    
    plt.tight_layout()
    if output_path:
        wandb_project = "local" if not wandb_project else wandb_project
        if not os.path.exists(os.path.join(output_path, wandb_project)):
            os.makedirs(os.path.join(output_path, wandb_project), exist_ok=True)
    else:
        wandb_project = "local" if not wandb_project else wandb_project
        output_path = os.path.join("./outputs", wandb_project)
        os.makedirs(output_path, exist_ok=True)

    output_filename = os.path.join(output_path, output_filename)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    # Create a summary dictionary for the evaluation metrics.
    metrics_dict = {"rmse": rmse_list, "mae": mae_list, "r2": r2_list}
    
    # Print summary report.
    print("Evaluation Metrics per Target:")
    for i in range(n_targets):
        print(f"Target {i+1}: RMSE = {rmse_list[i]:.2f}, MAE = {mae_list[i]:.2f}, R² = {r2_list[i]:.2f}")
    print(f"\nCombined evaluation plot saved to: {output_filename}")
    
    return metrics_dict

def extract_true_targets(dataloader):
    true_targets = []
    for i in range(len(dataloader.dataset.datasets[0])):
        true_targets.append(dataloader.dataset.datasets[0][i].y)
    
    return np.array(true_targets)