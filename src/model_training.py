import torch
import xgboost as xgb
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Initialize model with specified parameters, training data, and validation data for early stopping
def initialize_model(params, train_data, train_labels, weights, val_data=None, val_labels=None):
    model = xgb.XGBClassifier(**params)
    eval_set = [(train_data, train_labels)]  # Always include training data for evaluation
    if val_data is not None and val_labels is not None:
        eval_set.append((val_data, val_labels))  # Add validation set if provided
    
    model.fit(
        train_data,
        train_labels,
        sample_weight=weights,
        eval_set=eval_set,
        verbose=params.get('verbosity', 1)  # Set verbosity as per params
    )
    return model

# Save the trained model to the specified path
def save_model(model, path):
    joblib.dump(model, path)

# Calculate class weights for balanced training, to handle class imbalance
def calculate_class_weights(labels):
    unique_labels = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=labels)
    return {label: weight for label, weight in zip(unique_labels, weights)}

# Train model with specified parameters, training/validation data, and custom model path
def train_model(params, train_data, train_labels, val_data, val_labels, model_path, seed=42):
    # Calculate class weights based on training labels
    weights_dict = calculate_class_weights(train_labels)
    sample_weights = np.array([weights_dict[label] for label in train_labels])
    
    # Initialize and train the model, providing validation data for early stopping
    model = initialize_model(params, train_data, train_labels, sample_weights, val_data, val_labels)
    
    # Save the model to the specified model path
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model
