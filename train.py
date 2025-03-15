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

# Hyperparameters
BATCH_SIZE = 16  # Smaller batch size
EPOCHS = 10
LEARNING_RATE = 0.0001  # Reduced learning rate

# Load dataset
dataset = MUSANDataset(musan_dir="D:/Backup/musan")
print(f"Total dataset size: {len(dataset)}")

# Ensure dataset is not empty
if len(dataset) == 0:
    raise ValueError("Dataset is empty. Check musan_dir path and dataset files.")

# Split dataset
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Ensure train/val/test sizes are valid
if train_size == 0 or val_size == 0 or test_size == 0:
    raise ValueError("One of the dataset splits is zero. Consider increasing dataset size.")

print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

def collate_fn(batch):
    noisy_mels, mask, info = zip(*batch)
    noisy_mels = torch.stack(noisy_mels)
    mask = torch.stack(mask)
    return noisy_mels, mask, info

train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Define CNN + LSTM Model
class CNN_LSTM(nn.Module):
    def __init__(self, n_mels=128, hidden_size=128, num_layers=2, max_frames=186):
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
        self.upsample = nn.Upsample(size=(n_mels, max_frames), mode='bilinear', align_corners=False)
        self.final_activation = nn.Sigmoid()  # Thêm sigmoid để normalize output về [0,1]

    def forward(self, x):
        batch_size = x.size(0)

        # Ensure input has correct time dimension
        if x.size(3) != self.max_frames:
            x = nn.functional.pad(x, (0, self.max_frames - x.size(3), 0, 0), mode='constant', value=0)

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

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM().to(device)
criterion = nn.BCELoss()  # Use BCE loss thay vì MSE vì đã normalize data về [0,1]
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model():
    # Set up MLflow experiment
    mlflow.set_experiment("Audio_Noise_Detection")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "model_type": "CNN_LSTM",
            "hidden_size": 128,
            "num_layers": 2
        })

        best_val_loss = float('inf')

        for epoch in range(EPOCHS):
            # Training phase
            model.train()
            total_loss = 0

            # Use tqdm for progress tracking
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

            for batch_idx, (noisy_mel, mask_mel, info) in enumerate(progress_bar):
                noisy_mel, mask_mel = noisy_mel.to(device), mask_mel.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(noisy_mel)

                # Calculate loss
                loss = criterion(outputs, mask_mel)

                # Backward pass
                loss.backward()

                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Update metrics
                total_loss += loss.item()
                current_loss = total_loss / (batch_idx + 1)

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                })

            # Validation phase
            model.eval()
            val_loss = 0
            val_accuracy = 0
            total_samples = 0

            with torch.no_grad():
                for noisy_mel, mask_mel, _ in val_loader:
                    noisy_mel, mask_mel = noisy_mel.to(device), mask_mel.to(device)
                    outputs = model(noisy_mel)
                    val_loss += criterion(outputs, mask_mel).item()

                    # Calculate accuracy (threshold = 0.5)
                    predicted = (outputs > 0.5).float()
                    val_accuracy += (predicted == mask_mel).float().mean().item()
                    total_samples += 1

            # Calculate average metrics
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_accuracy / total_samples

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": avg_val_accuracy
            }, step=epoch)

            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Val Accuracy: {avg_val_accuracy:.4f}")

        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
        }, "models/best_model.pth")
        print("Saved new best model!")

        # Log final model to MLflow
        mlflow.pytorch.log_model(model, "final_model")
        print("Training complete!")

# Run training
if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    train_model()

# Load Model
def load_and_evaluate():
    try:
        checkpoint = torch.load("models/250316_cnn_lstm_loss_0.113_acc_0.9383.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_loss = 0
        test_accuracy = 0
        total_samples = 0

        with torch.no_grad():
            for noisy_mel, mask_mel, _ in tqdm(test_loader, desc="Evaluating"):
                noisy_mel, mask_mel = noisy_mel.to(device), mask_mel.to(device)
                outputs = model(noisy_mel)
                test_loss += criterion(outputs, mask_mel).item()

                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                test_accuracy += (predicted == mask_mel).float().mean().item()
                total_samples += 1

        avg_test_loss = test_loss / len(test_loader)
        avg_test_accuracy = test_accuracy / total_samples

        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {avg_test_accuracy:.4f}")

        return avg_test_loss, avg_test_accuracy

    except Exception as e:
        print(f"Error loading or evaluating model: {str(e)}")
        return None, None

# Evaluate model
load_and_evaluate()
