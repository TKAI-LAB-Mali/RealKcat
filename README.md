# RealKcat Repository

## Overview

The `RealKcat` repository provides tools and scripts to train and evaluate machine learning models for predicting enzyme kinetic parameters, specifically \( k_{cat} \) and \( K_M \), using various datasets. This repository is structured to include training and inference scripts for both `kcat` and `km` models, as well as utilities for data processing, model training, and standardized prediction.

## 📂 Download and Setup the Datasets

To retrain and reproduce results, please follow these steps to download and set up the required datasets correctly within the repository's `data` folder:

1. **Download the Dataset**:
   - Visit [https://chowdhurylab.github.io/downloads.html](https://chowdhurylab.github.io/downloads.html).
   - Locate **KinHub-27k (Manually-curated Enzyme Parameter Database; verified from 2158 papers)** and download the file (e.g., `KinHub-27k.zip`).

2. **Move the Downloaded File**:
   - Once downloaded, move the `KinHub-27k.zip` file to the `data` folder in the root directory of this repository.

3. **Extract the Files into the `data` Directory**:
   - Open a terminal or command prompt.
   - Navigate to the `data` directory:
     ```bash
     cd path/to/RealKcat/data
     ```
   - Unzip the dataset:
     - **On Linux/macOS**:
       ```bash
       unzip KinHub-27k.zip
       ```
     - **On Windows** (using Command Prompt or Git Bash):
       ```bash
       tar -xf KinHub-27k.zip
       ```
     - Alternatively, on Windows, right-click `KinHub-27k.zip` and choose "Extract All..." to unzip directly into the `data` folder.

4. **Verify the Extracted Files**:
   - After extraction, ensure your `data` folder has the following structure:

     ```
     data/
     ├── data_split/
     ├── PafA_data/
     ├── Save_kinetic_bin_range.pkl
     ├── WT_MD_database_v1.xlsx
     ├── WT_MD_dataset.pt
     ├── WT_MD_dataset_wNeg.pt
     ```

   - Confirm that all files and folders are present as shown above for the scripts to access the data properly.

5. **Proceed with Training or Inference**:
   - Once the data is correctly set up, use the provided scripts to perform training and inference.

## Your Folder RealKcat should have this structure:

```plaintext
RealKcat/
├── data/
│   ├── PafA_data/
│   │   ├── PafA_1_test_dataset_2.pt
│   │   ├── PafA_1_test_positions_2.pt
│   │   └── PafA_1_test_kcat_km_2.pt
│   ├── data_split/
│   ├── Save_kinetic_bin_range.pkl
│   ├── WT_MD_database_v1.xlsx
│   ├── WT_MD_dataset.pt
│   └── WT_MD_dataset_wNeg.pt
├── outputs/
├── scripts/
│   ├── test_ood_kcat_predict.py
│   ├── test_ood_km_predict.py
│   ├── test_PafA_kcat_predict.py
│   ├── test_PafA_km_predict.py
│   ├── train_kcat_model.py
│   └── train_km_model.py
├── src/
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── model_training.py
│   └── utils.py
├── LICENSE
├── README.md
└── RealKcat_Inference.ipynb
```

### Key Directories and Files

- **data/**: Contains all datasets, including training and test data, as well as supplementary files (e.g., bin range statistics for kinetic parameters).
  - `PafA_data/`: Datasets specifically for testing on the PafA enzyme.
  - `WT_MD_*`: Datasets for wild-type (WT) and mutant datasets.
- **outputs/**: Directory for saving model outputs, trained models, and prediction results.
- **scripts/**: Contains scripts for training and testing models.
  - `train_kcat_model.py` and `train_km_model.py`: Scripts for training models to predict `kcat` and `km`, respectively.
  - `test_*`: Scripts to run inference on test datasets for both `kcat` and `km`.
- **src/**: Contains utility scripts for data processing, model training, evaluation, and general utilities.
  - `data_processing.py`: Functions for loading and preparing datasets, standardizing data, and handling tensor data.
  - `model_training.py`: Functions for initializing and training XGBoost models with hyperparameters.
  - `evaluation.py`: Functions to evaluate model performance.
  - `utils.py`: Utility functions for device selection, setting seeds, etc.
- **RealKcat_Inference.ipynb**: Jupyter notebook for interactive inference.
- **README.md**: This file, providing documentation for the repository.

## Installation

### Prerequisites

- Python 3.8 or higher (codes are compatible with Python 3.12)
- Required libraries are listed in `requirements.txt`.

### Install Dependencies

To set up your environment, clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/RealKcat.git
cd RealKcat
pip install -r requirements.txt
```

### Requirements

- `torch`: For handling tensor operations and GPU compatibility.
- `xgboost`: For model training.
- `pandas`, `numpy`: For data manipulation.
- `scikit-learn`: For metrics and utilities.
- `matplotlib`: For plotting results.

## Usage

### 1. Training Models

To train models for `kcat` and `km` using XGBoost:

```bash
# Train kcat model
python scripts/train_kcat_model.py

# Train km model
python scripts/train_km_model.py
```

The scripts load datasets, apply global standardization, calculate class weights to handle imbalances, and train models with specific hyperparameters.

### 2. Running Inference

Run inference on test datasets using the trained models. Each `test_*_predict.py` script loads a model and dataset, applies standardization, and makes predictions.

#### Out-of-Distribution (OOD) Inference

Evaluate the model's performance on an out-of-distribution (OOD) dataset to test its robustness.

```bash
# Predict kcat on OOD dataset
python scripts/test_ood_kcat_predict.py

# Predict km on OOD dataset
python scripts/test_ood_km_predict.py
```

#### PafA Enzyme Inference

Use the PafA dataset for detailed testing on a specific enzyme.

```bash
# Predict kcat on PafA dataset
python scripts/test_PafA_kcat_predict.py

# Predict km on PafA dataset
python scripts/test_PafA_km_predict.py
```

### 3. Running Inference on Jupyter Notebook

For interactive analysis, use `RealKcat_Inference.ipynb` to explore and visualize predictions for `kcat` and `km`.

## Data Processing and Utilities

- **data_processing.py**: Functions for loading datasets, standardizing data, and creating custom datasets using PyTorch `TensorDataset`.
  - **Functions**:
    - `apply_global_standardization_separate()`: Applies global standardization using predefined means and standard deviations.
    - `get_train_stats_separate()`: Calculates global mean and standard deviation for each feature group.

- **evaluation.py**: Functions for evaluating model predictions.
  - **Functions**:
    - `evaluate_model()`: Computes metrics like accuracy, precision, recall, F1-score, and MCC.
    - `plot_confusion_matrix()`: Generates and saves confusion matrices for visual evaluation.

- **model_training.py**: Functions for initializing and training models.
  - **Functions**:
    - `initialize_model()`: Sets up an XGBoost model with specified parameters.
    - `calculate_class_weights()`: Computes class weights for balanced training.
    - `train_model()`: Trains and saves the model with training and validation datasets.

- **utils.py**: Utility functions.
  - **Functions**:
    - `check_device()`: Checks and returns the available device (CPU or GPU).
    - `set_seed()`: Sets seeds for reproducibility.

## Examples

### Example: Training a `kcat` Model

```bash
python scripts/train_kcat_model.py
```

### Example: Running Inference on `kcat` for OOD and PafA Datasets

```bash
# Run OOD inference for kcat
python scripts/test_ood_kcat_predict.py

# Run PafA inference for kcat
python scripts/test_PafA_kcat_predict.py
```

## Results and Visualization

Model outputs, trained models, and prediction results are saved as figures in the `outputs/` directory.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
