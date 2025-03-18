import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import mlflow
from tqdm import tqdm
from numpy import genfromtxt
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime
import os
from preprocessing import N_FEATURES


#region HYPERPARAMETERS

BATCH_SIZE = 50
EPOCHS = 40
LEARNING_RATE = 0.0002
INPUT_CHANNELS = 5  # Changed from 1 to 5
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 2e-4  # Tăng L2 regularization

#endregion


#region DATA LOADING
from datasets.audio_dataset import AudioDataset

x_train = genfromtxt('data/train_data.csv', delimiter=',')
y_train = genfromtxt('data/train_labels.csv', delimiter=',')
x_test = genfromtxt('data/test_data.csv', delimiter=',')
y_test = genfromtxt('data/test_labels.csv', delimiter=',')

# One-hot encoding
y_train = np.eye(NUM_CLASSES)[y_train.astype(int)]
y_test = np.eye(NUM_CLASSES)[y_test.astype(int)]

# Reshape data - swap dimensions to match model's expected input
total_samples = x_train.shape[0]
x_train = x_train.reshape(total_samples, INPUT_CHANNELS, N_FEATURES)  # [batch, channels, features]
x_test = x_test.reshape(x_test.shape[0], INPUT_CHANNELS, N_FEATURES)

# Create datasets with normalization
train_dataset = AudioDataset(x_train, y_train, normalize=True, augment=True)  # Enable augmentation for training
test_dataset = AudioDataset(x_test, y_test, normalize=True, augment=False)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#endregion


#region MODEL
from models.hybrid_model import HybridCNNAttention

model = HybridCNNAttention(
    input_channels=INPUT_CHANNELS,
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE,
).to(DEVICE)

# Count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'\nTrainable parameters: {count_parameters(model):,}\n')
# print('Model architecture:')
# print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

#endregion


#region TRAINING
# Create visualization directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
viz_dir = os.path.join('training_visualization', timestamp)
os.makedirs(viz_dir, exist_ok=True)

# Lists to store metrics
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

with mlflow.start_run():
    mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
    mlflow.log_param("EPOCHS", EPOCHS)
    mlflow.log_param("LEARNING_RATE", LEARNING_RATE)
    mlflow.log_param("INPUT_CHANNELS", LEARNING_RATE)
    mlflow.log_param("NUM_CLASSES", NUM_CLASSES)
    mlflow.log_param("WEIGHT_DECAY", WEIGHT_DECAY)
    mlflow.log_param("DEVICE", DEVICE.type)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct = 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            correct += (outputs.argmax(1) == batch_y.argmax(1)).sum().item()
            progress_bar.set_postfix(loss=loss.item())

        train_accuracy = correct / len(train_loader.dataset)
        train_loss /= len(train_loader)

        # Evaluate on test set every epoch
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                correct += (outputs.argmax(1) == batch_y.argmax(1)).sum().item()

        test_accuracy = correct / len(test_loader.dataset)
        test_loss /= len(test_loader)

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Logging to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)

        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # After each epoch
        scheduler.step(train_accuracy)

    # Save final plots
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Final Training and Test Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Final Training and Test Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'final_metrics.png'))
    plt.close()

    # Save metrics to CSV for future reference
    np.savetxt(os.path.join(viz_dir, 'metrics.csv'),
               np.column_stack([train_losses, train_accuracies, test_losses, test_accuracies]),
               delimiter=',',
               header='train_loss,train_acc,test_loss,test_acc',
               comments='')

    # Save model
    torch.save(model.state_dict(), "models/model_unet_lstm.pth")
    mlflow.log_artifact("models/model_unet_lstm.pth")
    print("\nModel Saved.\n")

#endregion
