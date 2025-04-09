import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from critic import BasicCritic
from encoder import DenseEncoder
from decoder import DenseDecoder
from utils.audio_to_stft import audio_to_stft, stft_to_audio
from utils.text_conversion import text_to_bits, bits_to_text
from utils.bit_accuracy import compare_bits
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import librosa
import librosa.display
import soundfile as sf


def load_trained_models(model_path, data_depth=1, hidden_size=64, device='cuda'):
    """Load trained encoder and decoder models"""
    # Initialize models
    encoder = DenseEncoder(data_depth, hidden_size).to(device)
    decoder = DenseDecoder(data_depth, hidden_size).to(device)
    critic = BasicCritic(hidden_size).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    critic.load_state_dict(checkpoint['state_dict_critic'])
    encoder.load_state_dict(checkpoint['state_dict_encoder'])
    decoder.load_state_dict(checkpoint['state_dict_decoder'])

    # Set to eval mode
    encoder.eval()
    decoder.eval()

    return encoder, decoder

def process_audio(audio_path, normalize=True):
    """Process audio file to spectrogram format matching training data"""
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(CURRENT_DIR, "datasets", "processed")
    params_path = os.path.join(PROCESSED_DIR, 'train', 'stft_params.json')
    with open(params_path, 'r') as f:
        params = json.load(f)

    sr = params['sample_rate']
    n_fft = params['n_fft']
    target_frames = params['target_frames']
    target_freq_bins = params['target_freq_bins']
    max_duration = params['max_duration']

    # Get STFT segments using audio_to_stft
    segments, sr, n_fft, hop_length = audio_to_stft(
        audio_path,
        sr=sr,
        n_fft=n_fft,
        target_frames=target_frames,
        target_freq_bins=target_freq_bins,
        segment_len_sec=max_duration,
        max_segments=1  # Get first segment only
    )

    if len(segments) == 0:
        raise ValueError("Could not extract any segments from audio file")

    # Take first segment
    features = segments[0]  # Shape: [2, 360, 360]

    # Calculate magnitude as third channel
    magnitude = np.sqrt(features[0]**2 + features[1]**2)

    # Stack all channels
    features = np.stack([features[0], features[1], magnitude], axis=0)

    # Convert to tensor
    cover = torch.FloatTensor(features)

    # Normalize if requested
    if normalize:
        for i in range(3):
            mu = cover[i].mean()
            std = cover[i].std()
            cover[i] = (cover[i] - mu) / (std + 1e-8)

    # Add batch dimension
    cover = cover.unsqueeze(0)  # Shape: [1, 3, 360, 360]

    return cover, (sr, n_fft, hop_length)

def test_hiding(encoder, decoder, audio_path, message, data_depth, device='cuda', save_audio=True):
    """Test hiding text message in audio file"""
    # Load and process audio
    cover, (sr, n_fft, hop_length) = process_audio(audio_path)
    cover = cover.to(device)

    # Convert message to binary with error correction
    message_bits = text_to_bits(message)
    message_len = len(message_bits)
    print(f"Message bits:", message_bits[:32])
    print(f"Message length in bits: {message_len}")

    # Calculate capacity
    N, C, H, W = cover.size()
    total_bits = H * W * data_depth
    print(f"Maximum capacity in bits: {total_bits}")

    if message_len > total_bits:
        raise ValueError(f"Message too long! Max bits: {total_bits}, Message bits: {message_len}")

    # Create payload tensor - only use exact message length
    payload_bits = np.zeros(total_bits, dtype=np.float32)  # Change to float32
    payload_bits[:message_len] = message_bits

    # Reshape payload to match training format
    payload = torch.tensor(payload_bits, device=device).float()
    payload = payload.reshape(N, data_depth, H, W)

    # Generate marked audio
    with torch.no_grad():
        generated = encoder(cover, payload)

        # Save generated audio if requested
        if save_audio:
            # Create output directory if needed
            os.makedirs('output', exist_ok=True)

            # Get real and imaginary components from generated spectrogram
            generated_numpy = generated.cpu().squeeze().numpy() # (3, 360, 360)
            print(f"Generated shape: {generated_numpy.shape}")
            # Convert to audio using only real and imaginary components
            generated_stft = generated_numpy[:2]  # Only real and imaginary parts (2, 360, 360)
            audio_signal = stft_to_audio(
                generated_stft,
                hop_length=hop_length
            )

            output_path = os.path.join('output', 'marked_audio.wav')
            sf.write(output_path, audio_signal, sr)
            print(f"\nSaved marked audio to: {output_path}")

        decoded = decoder(generated)

        # print("\nDecoder raw output (logits):")
        # print(decoded)

        # Calculate mean loss across all elements
        decoder_loss = torch.binary_cross_entropy_with_logits(decoded, payload, reduction=1)
        decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()

        print(f"Decoder loss: {decoder_loss.item():.3f}")
        print(f"Decoder accuracy: {decoder_acc.item():.3f}")

        # decoded = decoded[:, :, 1:, :]
        # Apply threshold to decoded bits
        decoded_bits = (decoded > 0).int()

        # print(f"Decoded bits: {decoded_bits}")

        decoded_bits_str = ''.join(map(str, decoded_bits.flatten().tolist()))
        decoded_bits_str = decoded_bits_str[:message_len]

        print(f"Decoded bits: {decoded_bits_str[:32]}")

        # Compare original and decoded bits
        compare_bits(str(message_bits), decoded_bits_str)

        # Try decoding message
        decoded_message = bits_to_text(decoded_bits_str)
        print(f"\nOriginal message: {message}")
        print(f"Decoded message: {decoded_message}")

    # Visualize results
    fig = plt.figure(figsize=(15, 10))

    # Original spectrograms
    ax1 = plt.subplot(2, 2, 1)
    real_db_orig = librosa.amplitude_to_db(np.abs(cover.cpu().squeeze()[0]), ref=np.max)
    img1 = librosa.display.specshow(real_db_orig,
                                  y_axis='linear',
                                  x_axis='time',
                                  ax=ax1,
                                  cmap='magma')
    ax1.set_title('Original Real Part')
    fig.colorbar(img1, ax=ax1, format="%+2.f dB")

    ax2 = plt.subplot(2, 2, 2)
    imag_db_orig = librosa.amplitude_to_db(np.abs(cover.cpu().squeeze()[1]), ref=np.max)
    img2 = librosa.display.specshow(imag_db_orig,
                                  y_axis='linear',
                                  x_axis='time',
                                  ax=ax2,
                                  cmap='magma')
    ax2.set_title('Original Imaginary Part')
    fig.colorbar(img2, ax=ax2, format="%+2.f dB")

    # Marked spectrograms
    ax3 = plt.subplot(2, 2, 3)
    real_db_marked = librosa.amplitude_to_db(np.abs(generated.cpu().squeeze()[0]), ref=np.max)
    img3 = librosa.display.specshow(real_db_marked,
                                  y_axis='linear',
                                  x_axis='time',
                                  ax=ax3,
                                  cmap='magma')
    ax3.set_title('Marked Real Part')
    fig.colorbar(img3, ax=ax3, format="%+2.f dB")

    ax4 = plt.subplot(2, 2, 4)
    imag_db_marked = librosa.amplitude_to_db(np.abs(generated.cpu().squeeze()[1]), ref=np.max)
    img4 = librosa.display.specshow(imag_db_marked,
                                  y_axis='linear',
                                  x_axis='time',
                                  ax=ax4,
                                  cmap='magma')
    ax4.set_title('Marked Imaginary Part')
    fig.colorbar(img4, ax=ax4, format="%+2.f dB")

    fig.suptitle('Spectrogram Comparison: Original vs Marked')
    plt.tight_layout()
    plt.savefig('marked_spectrogram.png', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_path = 'models/DenseEncoder_DenseDecoder_0.962_2025-04-05_22h46m42.dat'
    # model_path = 'models/DenseEncoder_DenseDecoder_0.964_2025-04-05_22h22m10.dat'
    model_path = 'models/DenseEncoder_DenseDecoder_0.915_2025-04-09_12h08m20.dat'
    # model_path = 'models/DenseEncoder_DenseDecoder_0.913_2025-04-09_11h55m52.dat'
    data_depth = 2
    hidden_size = 128
    save_audio = True

    # Load models
    encoder, decoder = load_trained_models(
        model_path,
        data_depth=data_depth,
        hidden_size=hidden_size,
        device=device
    )

    # Test with text message
    # audio_path = "D:\\Backup\\musan\\music\\fma\\music-fma-0000.wav"
    audio_path = "D:\\Backup\\musan\\music\\fma-western-art\\music-fma-wa-0008.wav"
    message = "hello"

    marked_spectrogram = test_hiding(encoder, decoder, audio_path, message, data_depth, device, save_audio)
