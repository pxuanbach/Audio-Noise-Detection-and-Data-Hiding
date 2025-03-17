import os
import torch
import numpy as np
from tqdm import tqdm
from dataset import MUSANDataset
from train import CNN_LSTM, HybridLoss
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.utils.data as data
from matplotlib import pyplot as plt
import seaborn as sns
import torch.nn as nn
import librosa
from utils import visualize_spectrograms

# Hyperparameters
# BATCH_SIZE = 32
# HIDDEN_SIZE = 256
# NUM_LAYERS = 2
MAX_FRAMES = 312

BATCH_SIZE = 64  # Increased from 32
HIDDEN_SIZE = 512  # Increased from 256
NUM_LAYERS = 3  # Increased from 2


def collate_fn(batch):
    noisy_mels, mask, info = zip(*batch)
    noisy_mels = torch.stack(noisy_mels)
    mask = torch.stack(mask)
    return noisy_mels, mask, info


def visualize_random_samples(model, val_loader, device, num_samples=5):
    # Get random indices
    all_indices = list(range(len(val_loader.dataset)))
    random_indices = np.random.choice(all_indices, num_samples, replace=False)

    model.eval()
    with torch.no_grad():
        for idx in random_indices:
            # Get sample
            noisy_mel, true_mask, info = val_loader.dataset[idx]
            noisy_mel = noisy_mel.unsqueeze(0).to(device)  # Add batch dimension
            true_mask = true_mask.unsqueeze(0).to(device)

            # Get prediction
            predicted_mask = model(noisy_mel)

            # Convert predictions to numpy and remove extra dimensions
            predicted_mask = predicted_mask.cpu().numpy()
            predicted_mask = np.squeeze(predicted_mask)  # Remove batch and channel dims
            predicted_mask = (predicted_mask > 0.5).astype(np.float32)

            # Create visualization with 2x2 layout
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Plot true mask and mel
            visualize_spectrograms(
                noisy_mel[0].cpu(),
                true_mask[0].cpu(),
                output_path=None,
                ax=(ax1, ax2),
                title="Ground Truth"
            )

            # Plot predicted mask and mel
            visualize_spectrograms(
                noisy_mel[0].cpu(),
                predicted_mask if predicted_mask.ndim == 2 else predicted_mask[0],
                output_path=None,
                ax=(ax3, ax4),
                title="Prediction"
            )

            plt.tight_layout()

            # Save plot
            os.makedirs("validation_results/random_samples", exist_ok=True)
            output_path = f'validation_results/random_samples/sample_{idx}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Print audio info
            print(f"\nSample {idx}:")
            print(f"Speech file: {info['speech_file']}")
            print(f"Noise file: {info['noise_file']}")
            print(f"Saved visualization to {output_path}")

def validate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = MUSANDataset("./preprocessed_dataset2")

    # Split dataset with 60:20:20 ratio
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    _, val_set, _ = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    val_loader = data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Load model
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model checkpoint not found")

    # Initialize model first
    model = CNN_LSTM(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, max_frames=MAX_FRAMES).to(device)
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Validation metrics
    all_preds = []
    all_targets = []
    val_losses = []
    criterion = HybridLoss(alpha=0.7)

    print("Starting validation...")
    with torch.no_grad():
        for noisy_mel, mask_mel, _ in tqdm(val_loader):
            noisy_mel = noisy_mel.to(device)
            mask_mel = mask_mel.to(device)

            outputs = model(noisy_mel)
            loss, _ = criterion(outputs, mask_mel)
            val_losses.append(loss.item())

            # Convert predictions to binary
            preds = (outputs > 0.5).float()

            # Store predictions and targets
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(mask_mel.cpu().numpy().flatten())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average='binary'
    )
    avg_loss = sum(val_losses) / len(val_losses)

    # Print results
    print("\nValidation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Validation Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    os.makedirs("validation_results", exist_ok=True)
    plt.savefig('validation_results/val_confusion_matrix.png')
    plt.close()

    # Visualize random samples
    print("\nVisualizing random samples...")
    visualize_random_samples(model, val_loader, device, 10)

if __name__ == "__main__":
    validate_model()
