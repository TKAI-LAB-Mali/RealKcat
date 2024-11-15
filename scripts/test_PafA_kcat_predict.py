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
import pickle
from sklearn.metrics import accuracy_score
from src.data_processing import TensorDataset, apply_global_standardization_separate, get_train_stats_separate
from src.utils import check_device, set_seed

def dataset_to_tensors(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, labels = next(iter(loader))
    return data, labels



class PafA_kcat_Inference:
    def __init__(self, model_path, data_file_path, bin_stats_path, device=None, output_dir="outputs"):
        self.device = device if device else check_device()
        self.output_dir = output_dir
        self.model = joblib.load(model_path)
        self.data_file_path = data_file_path
        self.bin_stats_path = bin_stats_path
        self.X_PafA_tensor, self.y1_PafA_tensor, self.y2_PafA_tensor = None, None, None
        set_seed(42)
        
        # Load data and bin ranges
        self.load_data()
        self.load_bin_stats()

    def load_data(self):
        self.X_PafA_tensor, self.y1_PafA_tensor, self.y2_PafA_tensor = torch.load(self.data_file_path)
        self.X_PafA_tensor, self.y1_PafA_tensor = self.X_PafA_tensor.to(self.device), self.y1_PafA_tensor.to(self.device)
        self.dataset = TensorDataset(self.X_PafA_tensor, self.y1_PafA_tensor)  #kcat

    def load_bin_stats(self):
        with open(self.bin_stats_path, 'rb') as f:
            data = pickle.load(f)
        self.kcat_auto_log_bin_stats = data['kcat_auto_log_bin_stats']
        print('Loaded kcat_auto_log_bin_stats successfully')

    def standardize_data(self, global_mean_1, global_std_1, global_mean_2, global_std_2):
        self.dataset_std = apply_global_standardization_separate(self.dataset, global_mean_1, global_std_1, global_mean_2, global_std_2)

    def convert_to_numpy(self):
        X_data, y_data = dataset_to_tensors(self.dataset_std)
        return X_data.cpu().numpy(), y_data.cpu().numpy()

    def predict(self, X_data):
        return self.model.predict(X_data)

    def get_bin_ranges(self, predicted_bins):
        bin_ranges = []
        for bin_idx in predicted_bins:
            bin_range = self.kcat_auto_log_bin_stats.loc[self.kcat_auto_log_bin_stats['auto_log_bin'] == bin_idx, ['min', 'max']].values[0]
            bin_ranges.append(bin_range)
        return bin_ranges

    def plot_results(self, y_true, y_pred_bins, experimental_kcat, positions, highlight_position=164, mutant_labels=None):
        # Map predicted bins to kcat ranges
        predicted_kcat_bins = self.get_bin_ranges(y_pred_bins)
        predicted_kcat_min = [pred[0] for pred in predicted_kcat_bins]
        predicted_kcat_max = [pred[1] for pred in predicted_kcat_bins]
        predicted_kcat_mid = [(pred[0] + pred[1]) / 2 for pred in predicted_kcat_bins]

        # Filter indices where predictions match true labels
        # match_indices = [i for i, (pred, true) in enumerate(zip(y_pred_bins, y_true)) if pred == true]
        match_indices = [i for i, (pred, true) in enumerate(zip(y_pred_bins, y_pred_bins)) if pred == true]

        filtered_positions = [positions[i] for i in match_indices]
        filtered_experimental_kcat = [experimental_kcat[i] for i in match_indices]
        filtered_predicted_kcat_mid = [predicted_kcat_mid[i] for i in match_indices]
        
        # Ensure that yerr contains non-negative values
        filtered_lower_errors = [abs(mid - min_val) for min_val, mid in zip(predicted_kcat_min, filtered_predicted_kcat_mid)]
        filtered_upper_errors = [abs(max_val - mid) for max_val, mid in zip(predicted_kcat_max, filtered_predicted_kcat_mid)]
        filtered_yerr = [filtered_lower_errors, filtered_upper_errors]

        highlight_indices = [i for i, pos in enumerate(positions) if pos == highlight_position]

        if highlight_indices:
            # Separate data for highlighted position
            highlight_experimental_kcat = [experimental_kcat[i] for i in highlight_indices]
            highlight_predicted_kcat_min = [predicted_kcat_min[i] for i in highlight_indices]
            highlight_predicted_kcat_max = [predicted_kcat_max[i] for i in highlight_indices]
            highlight_predicted_kcat_mid = [predicted_kcat_mid[i] for i in highlight_indices]

            plt.figure(figsize=(10, 6))

            # Plot experimental kcat values
            plt.scatter(filtered_positions, filtered_experimental_kcat, color='#0072B2', label=fr'Exp. $k_{{cat}}$', zorder=2)

            # Plot prediction range as error bars
            plt.errorbar(filtered_positions, filtered_predicted_kcat_mid, yerr=filtered_yerr, fmt='none', ecolor='grey', elinewidth=2, capsize=4, label=fr'Pred. $k_{{cat}}$ range', zorder=1)

            # Highlight occurrences of specific position
            for i, (kcat, min_val, max_val, mid) in enumerate(zip(highlight_experimental_kcat, highlight_predicted_kcat_min, highlight_predicted_kcat_max, highlight_predicted_kcat_mid)):
                index = highlight_indices[i]
                mutant_label = mutant_labels.get(index, "R164") if mutant_labels else "R164"
                color = '#D55E00' if mutant_label == "R164A" else 'blue'

                plt.scatter(highlight_position, kcat, color=color, s=100, edgecolor='black', label=fr'Exp. $k_{{cat}}$ (Mutant: {mutant_label})', zorder=3)
                plt.errorbar([highlight_position], [mid], yerr=[[mid - min_val], [max_val - mid]], fmt='none', ecolor=color, elinewidth=4, capsize=4, label=fr'Pred. $k_{{cat}}$ range (Mutant: {mutant_label})', zorder=3)

                plt.annotate(f"Catalytic residue\n({mutant_label})", (highlight_position, kcat), xytext=(90 if mutant_label == "R164A" else -70, -100), textcoords='offset points', arrowprops=dict(arrowstyle="->", color=color, lw=2.5), color='black', fontsize=12, ha='center', va='center')

            # Plot settings
            plt.yscale("log")
            plt.xticks(fontsize=15, fontname='Arial', color='black')
            plt.yticks(fontsize=15, fontname='Arial', color='black')
            plt.legend(fontsize=10, frameon=False, loc='best')
            plt.xlabel('Amino-acid residue position', fontsize=14, fontname='Arial', color='black')
            plt.ylabel('$k_{cat}$ values (log scale, $\\text{s}^{-1}$)', fontsize=14, fontname='Arial', color='black')
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2.0)
            ax.spines['bottom'].set_linewidth(2.0)
            plt.savefig(f'{self.output_dir}/PafA_kcat_predictions.png')
            plt.close()
            # plt.show()
        else:
            print(f"Position {highlight_position} not found in positions.")


# Usage example
if __name__ == "__main__":
    model_path = "outputs/kcat_model.pkl"
    data_file_path = './data/PafA_data/PafA_1_test_dataset_2.pt'
    bin_stats_path = './data/Save_kinetic_bin_range.pkl'

    # Load device
    device = check_device()
    print("Using device:", device)

    # Initialize inference
    inference = PafA_kcat_Inference(model_path=model_path, data_file_path=data_file_path, bin_stats_path=bin_stats_path, device=device)

    # Standardize data
    global_mean_1 = torch.tensor(-0.0006011285004206002, device=device)
    global_std_1 = torch.tensor(0.18902993202209473, device=device)
    global_mean_2 = torch.tensor(-0.00015002528380136937, device=device)
    global_std_2 = torch.tensor(0.6113553047180176, device=device)
    inference.standardize_data(global_mean_1, global_std_1, global_mean_2, global_std_2)

    # Convert to numpy and predict
    X_data, y_data = inference.convert_to_numpy()
    y_pred = inference.predict(X_data)

    # Load experimental values and positions
    test_positions = torch.load('./data/PafA_data/PafA_1_test_positions_2.pt').tolist()
    test_kcat = torch.load('./data/PafA_data/PafA_1_test_kcat_km_2.pt')[0].tolist()

    # Plot results
    mutant_labels = {123: "R164A", 200: "R164G"}
    inference.plot_results(y_true=y_data, y_pred_bins=y_pred, experimental_kcat=test_kcat, positions=test_positions, highlight_position=164, mutant_labels=mutant_labels)
