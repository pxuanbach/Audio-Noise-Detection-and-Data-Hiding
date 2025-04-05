import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import gc
from pathlib import Path
import numpy as np

from datasets.audio_dataset import AudioDataset
from encoder import DenseEncoder
from decoder import DenseDecoder
from critic import BasicCritic

# Setup logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class AudioSteganographyGAN:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        # Initialize models
        self.encoder = DenseEncoder(
            data_depth=config['data_depth'],
            hidden_size=config['hidden_size']
        ).to(self.device)

        self.decoder = DenseDecoder(
            data_depth=config['data_depth'],
            hidden_size=config['hidden_size']
        ).to(self.device)

        self.critic = BasicCritic(
            hidden_size=config['hidden_size']
        ).to(self.device)

        # Initialize optimizers
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=config['learning_rate']
        )
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(),
            lr=config['learning_rate']
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config['learning_rate']
        )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Initialize best losses for model saving
        self.best_encoder_loss = float('inf')

        # Add gradient clipping
        self.grad_clip = config.get('grad_clip', 1.0)

        # Enable gradient scaler for mixed precision
        self.scaler = torch.amp.GradScaler()

        # Add accuracy threshold
        self.accuracy_threshold = 0.5

        # Critic iterations
        self.critic_iters = 5  # Train critic ít hơn

    def calculate_accuracy(self, predictions, targets):
        """Calculate binary accuracy"""
        pred_labels = (predictions > self.accuracy_threshold).float()
        return (pred_labels == targets).float().mean().item()

    def train(self, train_loader):
        """Training loop based on fit_gan approach"""
        torch.cuda.empty_cache()

        for epoch in range(self.config['epochs']):
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            metrics = {'train.encoder_mse': [], 'train.decoder_loss': [],
                      'train.decoder_acc': [], 'train.cover_score': [],
                      'train.generated_score': []}

            critic_iter = 0

            for i, (cover, _) in enumerate(tqdm(train_loader)):
                # Clear memory mỗi 100 iterations
                if i % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                with torch.amp.autocast(device_type=self.device.type):  # Enable mixed precision
                    cover = cover.to(self.device, non_blocking=True)
                    N, _, H, W = cover.size()
                    payload = torch.zeros((N, self.config['data_depth'], H, W),
                                      device=self.device).random_(0, 2)

                    # Train critic mỗi critic_iters lần
                    if critic_iter < self.critic_iters:
                        generated = self.encoder(cover, payload)
                        cover_score = self.critic(cover).mean()
                        fake_score = self.critic(generated.detach()).mean()

                        critic_loss = -(cover_score - fake_score)
                        self.critic_optimizer.zero_grad(set_to_none=True)
                        self.scaler.scale(critic_loss).backward()
                        self.scaler.step(self.critic_optimizer)
                        self.scaler.update()

                        # Clamp weights
                        for p in self.critic.parameters():
                            p.data.clamp_(-0.1, 0.1)

                        # Update metrics for critic training
                        metrics['train.cover_score'].append(cover_score.item())
                        metrics['train.generated_score'].append(fake_score.item())

                        critic_iter += 1
                    else:
                        # Train encoder-decoder
                        generated = self.encoder(cover, payload)
                        decoded = self.decoder(generated)
                        fake_score = self.critic(generated).mean()

                        encoder_mse = self.mse_loss(generated, cover)
                        decoder_loss = self.bce_loss(decoded, payload)
                        decoder_acc = self.calculate_accuracy(decoded, payload)

                        total_loss = 100 * encoder_mse + decoder_loss - fake_score

                        self.encoder_optimizer.zero_grad(set_to_none=True)
                        self.decoder_optimizer.zero_grad(set_to_none=True)
                        self.scaler.scale(total_loss).backward()
                        self.scaler.step(self.encoder_optimizer)
                        self.scaler.step(self.decoder_optimizer)
                        self.scaler.update()

                        # Update metrics for encoder-decoder training
                        metrics['train.encoder_mse'].append(encoder_mse.item())
                        metrics['train.decoder_loss'].append(decoder_loss.item())
                        metrics['train.decoder_acc'].append(decoder_acc)
                        metrics['train.generated_score'].append(fake_score.item())

                        critic_iter = 0

            # Print epoch results
            print('encoder_mse: %.3f - decoder_loss: %.3f - decoder_acc: %.3f - cover_score: %.3f - generated_score: %.3f'
                  %(np.mean(metrics['train.encoder_mse']),
                    np.mean(metrics['train.decoder_loss']),
                    np.mean(metrics['train.decoder_acc']),
                    np.mean(metrics['train.cover_score']),
                    np.mean(metrics['train.generated_score'])))

            # Save model periodically
            if (epoch + 1) % self.config.get('save_interval', 5) == 0:
                self.save_models(f'model_epoch_{epoch+1}')

    def save_models(self, prefix):
        save_dir = Path('models')
        save_dir.mkdir(exist_ok=True)

        torch.save(self.encoder.state_dict(), save_dir / f'{prefix}_encoder.pth')
        torch.save(self.decoder.state_dict(), save_dir / f'{prefix}_decoder.pth')
        torch.save(self.critic.state_dict(), save_dir / f'{prefix}_critic.pth')

        # Save training config
        with open(save_dir / f'{prefix}_config.txt', 'w') as f:
            for key, value in self.config.items():
                f.write(f'{key}: {value}\n')

if __name__ == '__main__':
    # Training configuration
    config = {
        'data_depth': 1,          # Depth of secret message
        'hidden_size': 64,        # Number of hidden channels
        'learning_rate': 1e-4,
        'batch_size': 8,         # Reduced batch size
        'epochs': 10,
        'input_size': 360,        # STFT size
        'grad_clip': 1.0,         # Added gradient clipping
        'save_interval': 5        # Save model every 5 epochs
    }

    # Enable memory efficient options
    torch.backends.cudnn.benchmark = True

    # Initialize train dataset
    train_dataset = AudioDataset('datasets/processed/train', normalize=True)

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
    )

    # Initialize and train model
    model = AudioSteganographyGAN(config)
    model.train(train_loader)
