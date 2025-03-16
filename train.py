import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import mlflow
import numpy as np
from tqdm import tqdm
from dataset import MUSANDataset
import math
import os
from datetime import datetime
from utils import visualize_spectrograms
from matplotlib import pyplot as plt
import librosa

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_WORKERS = 4
MAX_FRAMES = 187  # Updated max_frames


def collate_fn(batch):
    noisy_mels, mask, info = zip(*batch)
    noisy_mels = torch.stack(noisy_mels)
    mask = torch.stack(mask)
    return noisy_mels, mask, info


# Define CNN + LSTM Model
class CNN_LSTM(nn.Module):
    def __init__(self, n_mels=128, hidden_size=128, num_layers=2, max_frames=187):  # Updated max_frames
        super(CNN_LSTM, self).__init__()

        self.max_frames = max_frames

        # Add batch normalization to CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate sizes after CNN
        self.n_mels_conv = n_mels // 4
        self.time_dim_conv = max_frames // 4

        self.lstm = nn.LSTM(input_size=(n_mels//4)*32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_mels)
        # Match input size exactly in upsampling
        self.upsample = nn.Upsample(size=(n_mels, self.max_frames), mode='bilinear', align_corners=False)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        current_frames = x.size(3)

        # Ensure input has correct time dimension
        if current_frames != self.max_frames:
            x = nn.functional.pad(x, (0, self.max_frames - current_frames, 0, 0), mode='constant', value=0)

        x = self.cnn(x)  # [batch, 32, n_mels/4, time_frames/4]
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch, time_frames/4, 32, n_mels/4]
        x = x.view(batch_size, self.time_dim_conv, -1)  # [batch, time_frames/4, feature_size]

        x, _ = self.lstm(x)
        x = self.fc(x)  # [batch, time_frames/4, n_mels]

        # Reshape and upsample to match target dimensions
        x = x.permute(0, 2, 1).contiguous().unsqueeze(1)  # [batch, 1, n_mels, time_frames/4]
        x = self.upsample(x)  # [batch, 1, n_mels, max_frames]
        x = self.final_activation(x)  # Normalize output

        return x

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7):  # Increase weight of BCE loss
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        # Add weight to reduce false positives
        bce_loss = self.bce(outputs, targets)

        # Penalize over-prediction of noise more heavily
        false_positive_weight = 2.0
        false_positive_loss = torch.mean(outputs * (1 - targets)) * false_positive_weight

        mse_loss = self.mse(outputs, targets)

        loss = self.alpha * (bce_loss + false_positive_loss) + (1 - self.alpha) * mse_loss

        return loss, {
            'bce_loss': bce_loss.item(),
            'mse_loss': mse_loss.item(),
            'fp_loss': false_positive_loss.item()
        }


def train_model(model, train_loader, val_loader):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = "noise_detection"
    mlflow.set_experiment(experiment_name)

    best_val_loss = float('inf')

    # Lists to store metrics history
    train_history = {
        'loss': [],
        'bce_loss': [],
        'mse_loss': []
    }
    val_history = {
        'loss': [],
        'bce_loss': [],
        'mse_loss': []
    }

    with mlflow.start_run(run_name=timestamp):
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "model_type": "CNN_LSTM",
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS
        })

        try:
            for epoch in range(EPOCHS):
                # Training phase
                model.train()
                train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
                train_losses = []
                train_bce_losses = []
                train_mse_losses = []

                for noisy_mel, mask_mel, _ in train_progress:
                    noisy_mel = noisy_mel.to(device, non_blocking=True)
                    mask_mel = mask_mel.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(noisy_mel)
                    loss, loss_components = criterion(outputs, mask_mel)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_losses.append(loss.item())
                    train_bce_losses.append(loss_components["bce_loss"])
                    train_mse_losses.append(loss_components["mse_loss"])

                    train_progress.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'bce': f'{loss_components["bce_loss"]:.4f}',
                        'mse': f'{loss_components["mse_loss"]:.4f}'
                    })

                # Calculate and store average losses for this epoch
                avg_loss = sum(train_losses) / len(train_losses)
                avg_bce = sum(train_bce_losses) / len(train_bce_losses)
                avg_mse = sum(train_mse_losses) / len(train_mse_losses)

                train_history['loss'].append(avg_loss)
                train_history['bce_loss'].append(avg_bce)
                train_history['mse_loss'].append(avg_mse)

                print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
                mlflow.log_metric("loss", avg_loss, step=epoch+1)

                # Validation phase
                model.eval()
                val_losses = []
                val_bce_losses = []
                val_mse_losses = []
                with torch.no_grad():
                    for noisy_mel, mask_mel, _ in val_loader:
                        noisy_mel = noisy_mel.to(device)
                        mask_mel = mask_mel.to(device)

                        outputs = model(noisy_mel)
                        loss, loss_components = criterion(outputs, mask_mel)
                        val_losses.append(loss.item())
                        val_bce_losses.append(loss_components["bce_loss"])
                        val_mse_losses.append(loss_components["mse_loss"])

                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_bce = sum(val_bce_losses) / len(val_bce_losses)
                avg_val_mse = sum(val_mse_losses) / len(val_mse_losses)

                val_history['loss'].append(avg_val_loss)
                val_history['bce_loss'].append(avg_val_bce)
                val_history['mse_loss'].append(avg_val_mse)

                print(f"Validation Loss: {avg_val_loss:.4f}")
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch+1)

                # Save model artifact in MLflow
                model_path = f"models/model_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path, artifact_path="models")

                # Save best model separately
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = "models/best_model.pth"
                    torch.save(model.state_dict(), best_model_path)
                    mlflow.log_artifact(best_model_path, artifact_path="best_model")

                # Log model architecture and hyperparameters
                mlflow.pytorch.log_model(
                    model,
                    f"model_epoch_{epoch+1}",
                    registered_model_name=f"noise_detection_model"
                )

        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print(f"Error during training: {str(e)}")

        # Plot training history
        epochs = range(1, len(train_history['loss']) + 1)

        plt.figure(figsize=(10, 6))

        # Plot all losses on same figure
        plt.plot(epochs, train_history['loss'], 'b-', label='Train Total Loss')
        plt.plot(epochs, val_history['loss'], 'b--', label='Val Total Loss')
        plt.plot(epochs, train_history['bce_loss'], 'r-', label='Train BCE Loss')
        plt.plot(epochs, val_history['bce_loss'], 'r--', label='Val BCE Loss')
        plt.plot(epochs, train_history['mse_loss'], 'g-', label='Train MSE Loss')
        plt.plot(epochs, val_history['mse_loss'], 'g--', label='Val MSE Loss')

        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Save plot
        os.makedirs("training_results", exist_ok=True)
        plt.savefig('training_results/loss_history.png')
        plt.close()


# Run training
if __name__ == "__main__":
    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = MUSANDataset(preprocessed_dir="./preprocessed_dataset")
    print()


    # Split dataset with 60:20:20 ratio
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    print(
        f"Total dataset: {len(dataset)},",
        f"Train set: {len(train_set)},",
        f"Validation set: {len(val_set)},",
        f"Test set: {len(test_set)}"
    )

    # Create data loaders
    train_loader = data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    model = CNN_LSTM(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, max_frames=MAX_FRAMES).to(device)
    criterion = HybridLoss(alpha=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    train_model(model, train_loader, val_loader)  # Pass val_loader
    print("Training complete!")
