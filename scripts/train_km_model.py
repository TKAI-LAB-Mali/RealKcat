import sys
import os

# Add the repository root to the Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

import torch
from src.data_processing import load_and_prepare_datasets, apply_global_standardization_separate, get_train_stats_separate
from src.model_training import train_model
from src.evaluation import calculate_metrics, calculate_e_accuracy, plot_confusion_matrix, plot_tsne
from src.utils import check_device, set_seed
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend


# Clear GPU cache
torch.cuda.empty_cache()  

# Define directories
model_code = 'outputs'
result_filename_csv = f"{model_code}/KM_results.xlsx"
save_dir = './data/data_split'

# Check if GPU is available
device = check_device()
print("Using device:", device)

# Set random seed for reproducibility
seed = 42
set_seed(seed)

# Define paths to dataset files
train_path = f'{save_dir}/train_dataset_y2_wPafA.pt'
val_path = f'{save_dir}/val_dataset_y2_wPafA.pt'
test_path = f'{save_dir}/test_dataset_y2.pt'

# Step 1: Load Data # Load datasets
train_dataset, val_dataset, test_dataset = load_and_prepare_datasets(train_path, val_path, test_path, device)

# Step 2: Standardize Data Using Training Stats
global_mean_1, global_std_1, global_mean_2, global_std_2 = get_train_stats_separate(train_dataset)
train_dataset = apply_global_standardization_separate(train_dataset, global_mean_1, global_std_1, global_mean_2, global_std_2)
val_dataset = apply_global_standardization_separate(val_dataset, global_mean_1, global_std_1, global_mean_2, global_std_2)
test_dataset = apply_global_standardization_separate(test_dataset, global_mean_1, global_std_1, global_mean_2, global_std_2)

# Step 3: Convert Data to PyTorch Tensors
def dataset_to_tensors(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, labels = next(iter(loader))
    return data, labels

train_data, train_labels = dataset_to_tensors(train_dataset)
val_data, val_labels = dataset_to_tensors(val_dataset)
test_data, test_labels = dataset_to_tensors(test_dataset)


# Define the best hyperparameters
best_params = {
    'n_estimators': 318,
    'max_depth': 50,
    'learning_rate': 0.0753493322967014,
    'max_delta_step': 4,
    'alpha': 1.230710427888436
}

# General model parameters
params = {
    **best_params,  # Unpack best_params directly into params
    'random_state': seed,
    'n_jobs': -1,
    'objective': 'multi:softmax',
    'num_class': len(torch.unique(train_labels)),
    'verbosity': 2,
    'eval_metric': ["mlogloss", "merror", "auc"],
    'early_stopping_rounds': 2,
    'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
    'subsample': 1
}

# Define output paths
os.makedirs("outputs", exist_ok=True)
model_path = "outputs/KM_model.pkl"  # Define model save path here
print("\nTraining the KM model...")
km_model = train_model(params, train_data.cpu().numpy(), train_labels.cpu().numpy(), val_data.cpu().numpy(), val_labels.cpu().numpy(), model_path=model_path, seed=seed)

# Step 5: Make Predictions and Evaluate Model
print("\nEvaluating the model...")
train_pred = km_model.predict(train_data.cpu().numpy())
val_pred = km_model.predict(val_data.cpu().numpy())
test_pred = km_model.predict(test_data.cpu().numpy())

# Calculate metrics for each dataset
train_metrics = calculate_metrics(train_labels.cpu().numpy(), train_pred)
val_metrics = calculate_metrics(val_labels.cpu().numpy(), val_pred)
test_metrics = calculate_metrics(test_labels.cpu().numpy(), test_pred)

# Calculate e-accuracy
train_e_accuracy = calculate_e_accuracy(train_labels.cpu().numpy(), train_pred)
val_e_accuracy = calculate_e_accuracy(val_labels.cpu().numpy(), val_pred)
test_e_accuracy = calculate_e_accuracy(test_labels.cpu().numpy(), test_pred)

# Print results
print("\nTrain Metrics:", train_metrics)
print("Train e-Accuracy:", train_e_accuracy)
print("\nValidation Metrics:", val_metrics)
print("Validation e-Accuracy:", val_e_accuracy)
print("\nTest Metrics:", test_metrics)
print("Test e-Accuracy:", test_e_accuracy)

# Step 6: Define output paths and save plots
os.makedirs("outputs", exist_ok=True)
confusion_matrix_path = "outputs/confusion_matrix_KM.png"
tsne_plot_path = "outputs/tsne_plot_KM.png"

print("\nSaving confusion matrix for test set...")
plot_confusion_matrix(test_labels.cpu().numpy(), test_pred, classes=range(len(torch.unique(train_labels))), output_path=confusion_matrix_path)

print("\nSaving t-SNE plot for test set...")
plot_tsne(test_data, test_labels.cpu().numpy(), test_pred, classes=range(len(torch.unique(train_labels))), output_path=tsne_plot_path)

# Step 7: Save Results
import pandas as pd
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "MCC", "AUC-PR", "e-Accuracy"],
    "Train": list(train_metrics) + [train_e_accuracy],
    "Validation": list(val_metrics) + [val_e_accuracy],
    "Test": list(test_metrics) + [test_e_accuracy]
})

results_df.to_excel(result_filename_csv, index=False)
print(f"\nResults saved to {result_filename_csv}")

torch.cuda.empty_cache()  # Final cache clearance