import torch
import torch.nn as nn
from encoder import DenseEncoder
from decoder import DenseDecoder
import torchaudio
import numpy as np
import os

def load_trained_models(model_path, data_depth=1, hidden_size=64, device='cuda'):
    """Load trained encoder and decoder models"""
    # Initialize models
    encoder = DenseEncoder(data_depth, hidden_size).to(device)
    decoder = DenseDecoder(data_depth, hidden_size).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint['state_dict_encoder'])
    decoder.load_state_dict(checkpoint['state_dict_decoder'])

    # Set to eval mode
    encoder.eval()
    decoder.eval()

    return encoder, decoder

def hide_message(encoder, audio_path, message, output_path, device='cuda'):
    """Hide a binary message in an audio file"""
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(audio_path)
    # Normalize to [-1, 1]
    waveform = waveform / torch.max(torch.abs(waveform))

    # Reshape audio to 2D (like image)
    audio_len = waveform.size(-1)
    height = int(np.sqrt(audio_len))
    width = int(np.ceil(audio_len / height))
    padding = height * width - audio_len

    # Pad if needed
    if padding > 0:
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    # Reshape to square
    cover = waveform.reshape(1, 1, height, width)

    # Prepare message
    payload = torch.tensor(message, device=device).float()
    payload = payload.reshape(1, 1, height, width)

    # Generate marked audio
    with torch.no_grad():
        cover = cover.to(device)
        marked = encoder(cover, payload)

    # Reshape back and save
    marked = marked.cpu().reshape(-1)[:audio_len]
    torchaudio.save(output_path, marked.unsqueeze(0), sample_rate)

    return marked

def extract_message(decoder, audio_path, device='cuda'):
    """Extract hidden message from marked audio"""
    # Load marked audio
    waveform, _ = torchaudio.load(audio_path)

    # Reshape to 2D
    audio_len = waveform.size(-1)
    height = int(np.sqrt(audio_len))
    width = int(np.ceil(audio_len / height))

    if height * width > audio_len:
        waveform = torch.nn.functional.pad(waveform, (0, height * width - audio_len))

    marked = waveform.reshape(1, 1, height, width)

    # Extract message
    with torch.no_grad():
        marked = marked.to(device)
        decoded = decoder(marked)

    # Convert to binary
    message = (decoded >= 0.0).cpu().numpy().reshape(-1)[:audio_len]

    return message

if __name__ == '__main__':
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'results/model/your_best_model.dat'  # Change to your model path

    # Load models
    encoder, decoder = load_trained_models(model_path, device=device)

    # Example usage
    audio_path = 'samples/test.wav'
    output_path = 'samples/marked.wav'

    # Create random message (for demo)
    message = np.random.randint(0, 2, size=44100)  # 1 second of binary data

    # Hide message
    print("Hiding message in audio...")
    marked_audio = hide_message(encoder, audio_path, message, output_path)

    # Extract message
    print("Extracting message from marked audio...")
    extracted_message = extract_message(decoder, output_path)

    # Calculate accuracy
    accuracy = np.mean(message == extracted_message)
    print(f"Message extraction accuracy: {accuracy:.2%}")
