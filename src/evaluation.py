import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

# Calculate common evaluation metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_true, y_pred)
    auc_pr = average_precision_score(label_binarize(y_true, classes=np.unique(y_true)), 
                                     label_binarize(y_pred, classes=np.unique(y_true)), 
                                     average="weighted")
    return accuracy, precision, recall, f1, mcc, auc_pr

# Calculate e-accuracy (within ±1 of the true label)
def calculate_e_accuracy(y_true, y_pred):
    within_range = ((y_pred >= (y_true - 1)) & (y_pred <= (y_true + 1))).sum()
    e_accuracy = within_range / len(y_true)
    return e_accuracy

# Plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, output_path="outputs/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(output_path)  # Save the figure to the specified output path
    plt.close()  # Close the figure to free up memory

# Plot and save t-SNE visualization of true labels vs. predictions
def plot_tsne(data, labels, predictions, classes, output_path="outputs/tsne_plot.png"):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data.cpu().numpy())
    plt.figure(figsize=(12, 6))

    # Plot true labels
    plt.subplot(1, 2, 1)
    for i in range(len(classes)):
        indices = np.where(labels == i)
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=f'Class {i}')
    plt.title('t-SNE visualization of true labels')
    plt.legend()

    # Plot predictions
    plt.subplot(1, 2, 2)
    for i in range(len(classes)):
        indices = np.where(predictions == i)
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=f'Class {i}')
    plt.title('t-SNE visualization of predictions')
    plt.legend()

    plt.savefig(output_path)  # Save the figure to the specified output path
    plt.close()  # Close the figure to free up memory
