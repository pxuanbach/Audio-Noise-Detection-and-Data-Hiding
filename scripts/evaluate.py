import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path

from datasets.audio_dataset import AudioDataset
from encoder import DenseEncoder
from decoder import DenseDecoder
from critic import BasicCritic

class ModelEvaluator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load config first
        self.config = self.load_config(model_path)

        # Initialize models with loaded config
        self.encoder = DenseEncoder(self.config['data_depth'], self.config['hidden_size']).to(self.device)
        self.decoder = DenseDecoder(self.config['data_depth'], self.config['hidden_size']).to(self.device)
        self.critic = BasicCritic(self.config['hidden_size']).to(self.device)

        # Load model weights
        self.load_models(model_path)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Metrics threshold
        self.accuracy_threshold = 0.5

    def load_config(self, model_path):
        """Load training configuration from saved config file"""
        path = Path(model_path)
        config = {}

        # Find the config file in the directory
        config_files = list(path.glob('*_config.txt'))
        if not config_files:
            raise FileNotFoundError(f"No config file found in {model_path}")

        config_file = config_files[0]  # Use first found config file

        # Parse config file
        with open(config_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                # Convert values to appropriate types
                try:
                    if '.' in value:
                        config[key] = float(value)
                    else:
                        config[key] = int(value)
                except ValueError:
                    config[key] = value  # Keep as string if conversion fails

        return config

    def load_models(self, path):
        path = Path(path)
        self.encoder.load_state_dict(torch.load(path / 'encoder.pth'))
        self.decoder.load_state_dict(torch.load(path / 'decoder.pth'))
        self.critic.load_state_dict(torch.load(path / 'critic.pth'))

    def evaluate(self, test_loader):
        """Full evaluation on test set"""
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()

        metrics = {
            'encoder_loss': [],
            'decoder_loss': [],
            'decoder_acc': [],
            'critic_acc': []
        }

        with torch.no_grad():
            for batch in test_loader:
                stft_data, _ = batch
                stft_data = stft_data.to(self.device)
                batch_size = stft_data.size(0)

                # Generate test messages
                secret = torch.randn(
                    batch_size,
                    self.config['data_depth'],
                    self.config['input_size'],
                    self.config['input_size']
                ).to(self.device)

                # Forward pass
                encoded_stft = self.encoder(stft_data, secret)
                decoded_secret = self.decoder(encoded_stft)
                critic_real = self.critic(stft_data)

                # Calculate metrics
                metrics['encoder_loss'].append(self.mse_loss(encoded_stft, stft_data).item())
                metrics['decoder_loss'].append(self.bce_loss(decoded_secret, secret).item())
                metrics['decoder_acc'].append(self.calculate_accuracy(decoded_secret, secret))
                metrics['critic_acc'].append(self.calculate_accuracy(
                    torch.sigmoid(critic_real),
                    torch.ones_like(critic_real)
                ))

        # Average metrics
        return {k: np.mean(v) for k, v in metrics.items()}

    def calculate_accuracy(self, predictions, targets):
        pred_labels = (predictions > self.accuracy_threshold).float()
        return (pred_labels == targets).float().mean().item()

if __name__ == '__main__':
    # Model path containing weights and config
    model_path = 'models/model_epoch_20'  # Adjust to your saved model path

    # Initialize test dataset
    test_dataset = AudioDataset('datasets/processed/test', normalize=True)

    # Initialize evaluator with model path
    evaluator = ModelEvaluator(model_path)

    # Create test dataloader using batch size from config
    test_loader = DataLoader(
        test_dataset,
        batch_size=evaluator.config.get('batch_size', 16),  # Use config batch size or default
        shuffle=False,
        num_workers=2
    )

    # Run evaluation and print results
    results = evaluator.evaluate(test_loader)
    print("\nEvaluation Results:")
    print(f"Model: {model_path}")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
