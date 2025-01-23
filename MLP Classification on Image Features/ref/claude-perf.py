# # def evaluate(
#     model: MLP,
#     test_loader: DataLoader,
#     criterion: torch.nn.Module,
#     device: torch.device,
# ) -> Tuple[float, float]:
#     num_items: int = len(test_loader.dataset)  # type: ignore
#     assert num_items > 0, "[evaluate] testing data must be present"
#     model.eval()
#     total_loss = 0
#     correct_preds = 0
#     with torch.no_grad():
#         for image, label in tqdm(test_loader, "evaluating model ... "):
#             image, label = image.to(device), label.to(device)
#             output = model(image)
#             loss = criterion(output, label)
#             total_loss += loss.item()
#             pred = output.argmax(dim=1)
#             correct_preds += (pred == label).sum().item()
#     accuracy = correct_preds / num_items
#     avg_loss = total_loss / num_items
#     return avg_loss, accuracy
#
# Add code to calculate any necessary metrics needed to calculate the accuracy, precision, f1 score, confusion matrix (perhaps num correct, num wrong, etc.), and add code to visualise these metrics in a table

from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate(
    model: MLP,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate model and compute comprehensive classification metrics.

    Args:
        model: Neural network model to evaluate
        test_loader: DataLoader containing test data
        criterion: Loss function
        device: Device to run evaluation on
        num_classes: Number of classes in the classification task

    Returns:
        Tuple of (average loss, metrics dictionary)
    """
    num_items: int = len(test_loader.dataset)  # type: ignore
    assert num_items > 0, "[evaluate] testing data must be present"

    model.eval()
    total_loss = 0

    # Store all predictions and labels for computing metrics
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for image, label in tqdm(test_loader, "evaluating model ... "):
            image, label = image.to(device), label.to(device)
            output = model(image)

            loss = criterion(output, label)
            total_loss += loss.item()

            pred = output.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    accuracy = (all_preds == all_labels).mean()
    avg_loss = total_loss / num_items

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Visualize metrics
    visualize_metrics(metrics, conf_matrix, num_classes)

    return avg_loss, metrics


def visualize_metrics(
    metrics: Dict[str, float], conf_matrix: np.ndarray, num_classes: int
) -> None:
    """
    Create visualizations for classification metrics and confusion matrix.
    """
    plt.figure(figsize=(15, 5))

    # Plot 1: Metrics Table
    plt.subplot(121)
    metrics_df = pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": [f"{v:.4f}" for v in metrics.values()],
        }
    )
    plt.axis("off")
    table = plt.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0.2, 0.2, 0.6, 0.6],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title("Classification Metrics")

    # Plot 2: Confusion Matrix
    plt.subplot(122)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.show()


# Example usage:
"""
test_loss, metrics = evaluate(model, test_loader, criterion, device, num_classes=10)
print(f"Test Loss: {test_loss:.4f}")
print("\nDetailed Metrics:")
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")
"""

# precision, recall, f1, * = precision*recall_fscore_support(
#         labels, preds, average="weighted"
#     )
#
# What does the average="weighted" param mean?

# The average="weighted" parameter in precision_recall_fscore_support() is important for multi-class classification. Here's a detailed breakdown:
# When calculating precision, recall, and F1 score in a multi-class classification problem, you have several averaging options:
#
# "weighted":
#
# Calculates metrics for each class
# Computes their average weighted by the number of true instances for each class
# Accounts for class imbalance
# Most useful when classes have different numbers of samples
#
#
# "macro":
#
# Simple average of metrics across all classes
# Gives equal weight to each class, regardless of its size
# Can be misleading if classes are imbalanced
#
#
# "micro":
#
# Calculates metrics globally by counting total true positives, false negatives, and false positives
# Gives equal weight to each instance
# Good for imbalanced datasets where you want to weight by number of instances
