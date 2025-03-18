import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np
import os
from datetime import datetime
from models.unet_lstm import UNetLSTM
from dataset import MusanNoiseDataset
from torch.utils.data import DataLoader
from transforms import MelSpectrogramTransform, LabelTransform

def create_output_dir():
    """Create output directory with current date"""
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join("trained_results", date_str)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot and save confusion matrix with percentage"""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues')
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_pred_prob, output_dir):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_prob, output_dir):
    """Plot and save Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()

def plot_prediction_examples(model, test_loader, output_dir, device, num_examples=5):
    """Plot example predictions vs ground truth"""
    model.eval()
    fig, axes = plt.subplots(num_examples, 2, figsize=(15, 4*num_examples))

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_examples:
                break

            noisy_mel = batch['noisy_mel'].float().to(device)
            labels = batch['labels'].float()

            outputs = model(noisy_mel)
            predictions = (outputs > 0.5).float().cpu()

            # Plot original mel spectrogram
            axes[i, 0].imshow(noisy_mel[0, 0].cpu().numpy(), aspect='auto', origin='lower')
            axes[i, 0].set_title(f'Input Mel Spectrogram {i+1}')

            # Plot predictions vs ground truth
            axes[i, 1].plot(labels[0].numpy(), label='Ground Truth', alpha=0.7)
            axes[i, 1].plot(predictions[0, 0].numpy(), label='Predictions', alpha=0.7)
            axes[i, 1].set_title(f'Predictions vs Ground Truth {i+1}')
            axes[i, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_examples.png'))
    plt.close()

def main():
    # Parameters
    BATCH_SIZE = 32
    MAX_LENGTH = 512
    MODEL_PATH = "models/best_model.pth"

    # Create output directory
    output_dir = create_output_dir()
    print(f"Saving results to: {output_dir}")

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load test dataset
    dataset = MusanNoiseDataset(
        dataset_dir="musan_dataset",
        transforms={
            'mel': MelSpectrogramTransform(max_length=MAX_LENGTH),
            'label': LabelTransform(max_length=MAX_LENGTH)
        }
    )

    # Split for test set (30%)
    test_size = int(0.3 * len(dataset))
    _, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Collect predictions
    all_labels = []
    all_preds = []
    all_pred_probs = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            noisy_mel = batch['noisy_mel'].float().to(device)
            labels = batch['labels'].float()

            outputs = model(noisy_mel)

            # Match dimensions
            min_length = min(outputs.size(-1), labels.size(-1))
            outputs = outputs[..., :min_length]
            labels = labels[..., :min_length]

            predictions = (outputs > 0.5).float().cpu()

            # Collect results with matched dimensions
            all_labels.extend(labels.numpy().flatten())
            all_preds.extend(predictions.squeeze(1).numpy().flatten())  # Remove channel dimension
            all_pred_probs.extend(outputs.squeeze(1).cpu().numpy().flatten())

    # Generate visualization plots
    print("Generating plots...")
    plot_confusion_matrix(all_labels, all_preds, output_dir)
    plot_roc_curve(all_labels, all_pred_probs, output_dir)
    plot_precision_recall_curve(all_labels, all_pred_probs, output_dir)
    plot_prediction_examples(model, test_loader, output_dir, device)

    print("Visualization complete!")

if __name__ == "__main__":
    main()
