import sys
import os

# Add the repository root to the Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score
from src.data_processing import TensorDataset, apply_global_standardization_separate, get_train_stats_separate
from src.utils import check_device, set_seed

def dataset_to_tensors(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, labels = next(iter(loader))
    return data, labels

class KMInference:
    def __init__(self, model_path, test_file_path, device=None, output_dir="outputs"):
        self.device = device if device else check_device()
        self.output_dir = output_dir
        self.model = joblib.load(model_path)  # Load the model
        self.test_file_path = test_file_path
        set_seed(42)
        
        # Load and prepare test data
        self.X_test_WT_tensor, self.X_test_MD_Ala_tensor, self.y1_test_WT_tensor, self.y2_test_WT_tensor = None, None, None, None
        self.load_test_data()

    def load_test_data(self):
        self.X_test_WT_tensor, self.X_test_MD_Ala_tensor, self.y1_test_WT_tensor, self.y2_test_WT_tensor = torch.load(self.test_file_path)
        self.X_test_WT_tensor, self.X_test_MD_Ala_tensor = self.X_test_WT_tensor.to(self.device), self.X_test_MD_Ala_tensor.to(self.device)
        self.y2_test_WT_tensor = self.y2_test_WT_tensor.to(self.device)

    def standardize_test_data(self, global_mean_1, global_std_1, global_mean_2, global_std_2):
        wt_dataset = TensorDataset(self.X_test_WT_tensor, self.y2_test_WT_tensor)
        md_dataset = TensorDataset(self.X_test_MD_Ala_tensor, self.y2_test_WT_tensor)
        self.testCatAware_WT_dataset_std = apply_global_standardization_separate(wt_dataset, global_mean_1, global_std_1, global_mean_2, global_std_2)
        self.testCatAware_MD_dataset_std = apply_global_standardization_separate(md_dataset, global_mean_1, global_std_1, global_mean_2, global_std_2)

    def convert_to_numpy(self):
        X_CatAware_WT_test_data, test_CatAware_WT_y2 = dataset_to_tensors(self.testCatAware_WT_dataset_std)
        X_CatAware_MD_test_data, test_CatAware_MD_y2 = dataset_to_tensors(self.testCatAware_MD_dataset_std)
        return X_CatAware_WT_test_data.cpu().numpy(), test_CatAware_WT_y2.cpu().numpy(), X_CatAware_MD_test_data.cpu().numpy(), test_CatAware_MD_y2.cpu().numpy()

    def predict(self, X_test_data):
        return self.model.predict(X_test_data)

    def evaluate(self, y_true, y_pred, label="WT"):
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy for {label} test set: {accuracy}")
        return accuracy

    def compare_predictions(self, y_true_wt, y_pred_wt, y_pred_md):
        comparison_df = pd.DataFrame({
            'True y2 (WT)': y_true_wt,
            'Predicted y2 (WT)': y_pred_wt,
            'Predicted y2 (MD)': y_pred_md
        })
        comparison_df['MD_lower_than_WT'] = comparison_df['Predicted y2 (MD)'] < comparison_df['Predicted y2 (WT)']
        comparison_df['MD_lower_than_WT_andTruePred'] = (comparison_df['MD_lower_than_WT'] &
                                                         (comparison_df['True y2 (WT)'] == comparison_df['Predicted y2 (WT)']))
        count_md_lower = comparison_df['MD_lower_than_WT'].sum()
        count_md_lower_and_true = comparison_df['MD_lower_than_WT_andTruePred'].sum()
        print(f"Cases where MD is lower than WT: {count_md_lower}")
        print(f"Cases where MD is lower and WT is correctly predicted: {count_md_lower_and_true}")
        return comparison_df

    def plot_comparisons(self, y_true, y_pred, y_md_pred):
        # Plot WT vs predicted WT
        plt.figure(figsize=(8, 6))
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], label='y=x', color='red', linestyle='--')
        plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label='Predictions vs True')
        for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
            plt.text(true_val, pred_val, str(i), fontsize=8, ha='right', va='bottom', color='black')
        plt.xlabel('True y2 (WT)')
        plt.ylabel('Predicted y2 (WT)')
        plt.title('True vs Predicted y2 (WT)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/WT_vs_Predictions_test_ood_KM.png')
        plt.close()

        # Plot WT vs MD predictions
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, y_md_pred, color='dodgerblue', alpha=0.7, s=100, label='MD vs WT Predictions')
        plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], label='y=x', color='red', linestyle='--')
        for i, (wt_pred, md_pred) in enumerate(zip(y_pred, y_md_pred)):
            plt.text(wt_pred, md_pred, str(i), fontsize=8, ha='right', va='bottom', color='black')
        plt.xlabel('Predicted y2 (WT)')
        plt.ylabel('Predicted y2 (MD)')
        plt.title('MD vs WT Predictions')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/MD_vs_WT_Predictions_test_ood_KM.png')
        plt.close()

    def display_prediction_ranges(self, predictions, class_ranges):
        for i, pred_class in enumerate(predictions):
            low, high = class_ranges[pred_class]["low"], class_ranges[pred_class]["high"]
            print(f"Sample {i + 1}: Predicted Class for KM = {pred_class}, Range for KM = [{low}, {high}]")

# Usage example
if __name__ == "__main__":
    # Configuration
    model_path = "outputs/KM_model.pkl"
    test_file_path = './data/data_split/test_ood_cataware.pt'
    
    class_ranges_y2 = {
        0: {"low": 1.0e-10, "high": 1.0e-5},    # Example values; adjust as needed
        1: {"low": 1.01e-5, "high": 1.0e-4},
        2: {"low": 1.002e-4, "high": 1.0e-3},
        3: {"low": 1.002e-3, "high": 1.0e-2},
        4: {"low": 1.008e-2, "high": 1.0e-1},
        5: {"low": 1.01e-1, "high": 1.02e2},
        }


   
    # Check if GPU is available
    device = check_device()
    print("Using device:", device)

    # Initialize inference
    inference = KMInference(model_path=model_path, test_file_path=test_file_path, device=device)

    # Standardize test data using training statistics, convert to tensors
    global_mean_1 = torch.tensor(-0.0006011285004206002, device=device)
    global_std_1 = torch.tensor(0.18902993202209473, device=device)
    global_mean_2 = torch.tensor(-0.00015002528380136937, device=device)
    global_std_2 = torch.tensor(0.6113553047180176, device=device)

    inference.standardize_test_data(global_mean_1, global_std_1, global_mean_2, global_std_2)

    # Convert to numpy arrays
    X_WT, y_WT, X_MD, y_MD = inference.convert_to_numpy()

    # Predictions
    y_pred_WT = inference.predict(X_WT)
    y_pred_MD = inference.predict(X_MD)

    # Evaluate and compare
    inference.evaluate(y_WT, y_pred_WT, label="WT")
    inference.evaluate(y_MD, y_pred_MD, label="MD")
    comparison_df = inference.compare_predictions(y_WT, y_pred_WT, y_pred_MD)

    # Plot comparisons and display ranges
    inference.plot_comparisons(y_WT, y_pred_WT, y_pred_MD)
    inference.display_prediction_ranges(y_pred_WT, class_ranges_y2)