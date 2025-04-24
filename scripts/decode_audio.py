import torch
import torchaudio
import numpy as np
from inference import load_trained_models, make_message
from utils.audio_to_stft import audio_to_stft
from datasets import SingleAudioLoader
from torchvision import transforms
import sys
import os

audio_path = sys.argv[1]

# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_path = 'models/gan_32_2_2_epochs_32/DenseEncoder_DenseDecoder_0.966_2025-04-21_16h21m37.dat'
base_dir = os.path.dirname(os.path.abspath(__file__))  # thư mục chứa script hiện tại
model_path = os.path.join(base_dir, 'models', 'gan_32_2_2_epochs_32', 'DenseEncoder_DenseDecoder_0.966_2025-04-21_16h21m37.dat')
channels_size = 2
data_depth = 2
hidden_size = 32

# Load models
encoder, decoder = load_trained_models(
    model_path,
    data_depth=data_depth,
    hidden_size=hidden_size,
    channels_size=channels_size,
    device=device
)

# Setup audio loading
# audio_path = "generated_reconstructed.wav"
transform = transforms.Compose([transforms.Lambda(lambda wav: audio_to_stft(wav))])
audio_loader = SingleAudioLoader(audio_path, transform=transform)

# Prepare features
cover, path, hop_length = audio_loader[0]
cover = cover[None].to(device)


# Add batch dimension and move to device
# cover = cover.unsqueeze(0).to(device)

# Decode message
decoded_message = make_message(cover, decoder=decoder, device=device)
print("Decoded message:", decoded_message)
