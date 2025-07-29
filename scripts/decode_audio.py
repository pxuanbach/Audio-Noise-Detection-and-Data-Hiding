import sys
import os
import torch
import torchaudio
import numpy as np
from inference import load_trained_models, make_message
from utils.audio_to_stft import audio_to_stft
from datasets import SingleAudioLoader
from torchvision import transforms

if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 2:
    # print("Usage: python decode_audio.py <audio_path>")
    sys.exit(1)

audio_path = sys.argv[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(base_dir, 'models', 'gan_32_2_2_epochs_32', 'DenseEncoder_DenseDecoder_0.966_2025-04-21_16h21m37.dat')
channels_size = 2
data_depth = 2
hidden_size = 32

encoder, decoder = load_trained_models(
    model_path,
    data_depth=data_depth,
    hidden_size=hidden_size,
    channels_size=channels_size,
    device=device
)

transform = transforms.Compose([transforms.Lambda(lambda wav: audio_to_stft(wav))])
audio_loader = SingleAudioLoader(audio_path, transform=transform)

cover, path, hop_length = audio_loader[0]
cover = cover[None].to(device)
# print(f"Cover shape: {cover.shape}, Device: {cover.device}, Min: {cover.min().item()}, Max: {cover.max().item()}")

decoded_message = make_message(cover, decoder=decoder, device=device)
print("Decoded message:", decoded_message)
# exec(decoded_message)