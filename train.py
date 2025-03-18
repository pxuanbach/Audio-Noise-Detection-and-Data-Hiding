import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import mlflow
import mlflow.pytorch
from dataset import MusanNoiseDataset
from models.unet_lstm import UNetLSTM
import os
from tqdm import tqdm

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.bce_weight = bce_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, outputs, targets):
        # Ensure targets have the same shape as outputs [batch, 1, time]
        if len(targets.shape) == 2:
            targets = targets.unsqueeze(1)  # Add channel dimension

        # # Ensure the time dimensions match
        # min_length = min(outputs.size(-1), targets.size(-1))
        # outputs = outputs[..., :min_length]
        # targets = targets[..., :min_length]

        mse_loss = self.mse(outputs, targets)
        bce_loss = self.bce(outputs, targets)

        return self.mse_weight * mse_loss + self.bce_weight * bce_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            noisy_mel = batch['noisy_mel'].float().to(device)
            labels = batch['labels'].float().to(device)

            # Forward pass
            outputs = model(noisy_mel)  # Shape: [batch, 1, time]

            # Ensure labels match output dimensions
            min_length = min(outputs.size(-1), labels.size(-1))
            outputs_trimmed = outputs[..., :min_length]
            labels_trimmed = labels[..., :min_length].unsqueeze(1)

            # Calculate loss and accuracy with matched dimensions
            loss = criterion(outputs_trimmed, labels_trimmed)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics with matched dimensions
            train_loss += loss.item()
            pred = (outputs_trimmed > 0.5).float()
            train_correct += (pred == labels_trimmed).sum().item()
            train_total += labels_trimmed.numel()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                noisy_mel = batch['noisy_mel'].float().to(device)
                labels = batch['labels'].float().to(device)

                outputs = model(noisy_mel)

                # Match dimensions for validation
                min_length = min(outputs.size(-1), labels.size(-1))
                outputs_trimmed = outputs[..., :min_length]
                labels_trimmed = labels[..., :min_length].unsqueeze(1)

                loss = criterion(outputs_trimmed, labels_trimmed)

                val_loss += loss.item()
                pred = (outputs_trimmed > 0.5).float()
                val_correct += (pred == labels_trimmed).sum().item()
                val_total += labels_trimmed.numel()

        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # Log metrics
        mlflow.log_metrics({
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        }, step=epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            mlflow.pytorch.log_model(model, "best_model")
            torch.save(model.state_dict(), os.path.join('models', 'best_model.pth'))

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')


def collate_fn(batch):
    """Simple collate function as padding is handled by transforms"""
    return {
        "noisy_mel": torch.stack([x["noisy_mel"] for x in batch]),  # Shape: [batch, 3, freq, time]
        "labels": torch.stack([x["labels"] for x in batch]),
        "metadata": [x["metadata"] for x in batch]
    }

def main():
    # Set hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    MAX_LENGTH = 512  # Fixed length for all spectrograms

    # Create save directory
    os.makedirs('models', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize transforms
    from transforms import MelSpectrogramTransform, LabelTransform

    # Load dataset with transforms
    dataset = MusanNoiseDataset(
        dataset_dir="musan_dataset",
        transforms={
            'mel': MelSpectrogramTransform(max_length=MAX_LENGTH),
            'label': LabelTransform(max_length=MAX_LENGTH)
        }
    )

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, collate_fn=collate_fn)

    # Initialize model and loss
    model = UNetLSTM().to(device)
    criterion = CombinedLoss(mse_weight=0.6, bce_weight=0.4)  # Adjust weights as needed
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start MLflow tracking
    mlflow.set_experiment("noise_detection")

    with mlflow.start_run():
        # Log parameters including loss weights
        mlflow.log_params({
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'optimizer': type(optimizer).__name__,
            'model_type': 'UNet-LSTM',
            'mse_weight': criterion.mse_weight,
            'bce_weight': criterion.bce_weight
        })

        # Train model
        train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)

if __name__ == '__main__':
    main()
