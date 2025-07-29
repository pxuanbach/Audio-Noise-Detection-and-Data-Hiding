import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from critic import BasicCritic
from datasets import AudioToImageFolder, SingleAudioLoader
from encoder import DenseEncoder
from decoder import DenseDecoder
from utils.audio_to_stft import audio_to_stft, stft_to_audio
from utils.text_conversion import bits_to_bytearray, bytearray_to_text, text_to_bits, bits_to_text
from utils.bit_accuracy import compare_bits
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import librosa
import librosa.display
import soundfile as sf
from collections import Counter


def load_trained_models(model_path, data_depth=1, hidden_size=64, channels_size=3, device='cuda'):
    """Load trained encoder and decoder models"""
    # Initialize models
    encoder = DenseEncoder(data_depth, hidden_size, channels_size).to(device)
    decoder = DenseDecoder(data_depth, hidden_size, channels_size).to(device)
    critic = BasicCritic(hidden_size, channels_size).to(device)

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


def make_payload(width, height, depth, text):
    """
    This takes a piece of text and encodes it into a bit vector. It then
    fills a matrix of size (width, height) with copies of the bit vector.
    """
    message = text_to_bits(text) + [0] * 32

    payload = message
    while len(payload) < width * height * depth:
        payload += message

    payload = payload[:width * height * depth]

    return torch.FloatTensor(payload).view(1, depth, height, width)


def test_audio(encoder, decoder, cover, payload, save_wav=True, hop_length=512, file_path=None):
    # get sample rate
    _, sr = torchaudio.load(file_path)
    print(f"Sample rate: {sr}")

    generated = encoder.forward(cover, payload)
    decoded = decoder.forward(generated)

    decoder_loss = torch.nn.functional.binary_cross_entropy_with_logits(decoded, payload)
    decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()

    print("Decoder loss: %.3f" % decoder_loss.item())
    print("Decoder acc: %.3f" % decoder_acc.item())

    f, ax = plt.subplots(1, 3, figsize=(16, 5))
    f.suptitle("%s_%s" % (encoder.name, decoder.name), fontsize=16)
    f.tight_layout(pad=4.0)

    def extract_complex_spec(tensor):
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.cpu().detach().numpy()
        complex_spec = tensor[0] + 1j * tensor[1]
        return complex_spec

    cover_spec = extract_complex_spec(cover)
    generated_spec = extract_complex_spec(generated)

    # Magnitudes (for display)
    cover_mag = np.abs(cover_spec)
    generated_mag = np.abs(generated_spec)

    # Plot Cover
    librosa.display.specshow(librosa.amplitude_to_db(cover_mag, ref=np.max),
                             x_axis='time', y_axis='linear',
                             sr=sr, hop_length=hop_length, ax=ax[0])
    ax[0].set_title('Cover Image')

    # Plot Generated
    librosa.display.specshow(librosa.amplitude_to_db(generated_mag, ref=np.max),
                             x_axis='time', y_axis='linear',
                             sr=sr, hop_length=hop_length, ax=ax[1])
    ax[1].set_title('Generated Image')

    # Plot Payload diff
    diff = cover_mag - generated_mag
    librosa.display.specshow(diff, x_axis='time', y_axis='linear',
                             sr=sr, hop_length=hop_length, ax=ax[2])
    ax[2].set_title('Payload (Difference)')

    plt.show()

    if save_wav:
        print("Converting and saving WAV files...")

        if len(cover.shape) == 4:
            _cover = cover.squeeze(0).cpu().detach().numpy()
        else:
            _cover = cover.cpu().detach().numpy()
        cover_audio = torch.tensor(stft_to_audio(_cover, hop_length=hop_length))[None, :]

        if len(generated.shape) == 4:
            _generated = generated.squeeze(0).cpu().detach().numpy()
        else:
            _generated = generated.cpu().detach().numpy()
        generated_audio = torch.tensor(stft_to_audio(_generated, hop_length=hop_length))[None, :]

        torchaudio.save("cover_reconstructed.wav", cover_audio, sr)
        torchaudio.save("generated_reconstructed.wav", generated_audio, sr)

        print("Saved: cover_reconstructed.wav & generated_reconstructed.wav")

    return generated

def make_message(image, decoder=None, device='cuda'):
    image = image.to(device)

    image = decoder(image).view(-1) > 0
    # image = torch.tensor(image, dtype=torch.uint8)
    image = image.clone().detach().to(dtype=torch.uint8)
    
    # split and decode messages
    candidates = Counter()
    bits = image.data.cpu().numpy().tolist()
    for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
      #print(candidate)
      candidate = bytearray_to_text(bytearray(candidate))
      if candidate:
          candidates[candidate] += 1

    # choose most common message
    if len(candidates) == 0:
      raise ValueError('Failed to find message.')

    candidate, count = candidates.most_common(1)[0]
    return candidate


def test_hiding(encoder, decoder, test_set_item, message, data_depth, device='cuda'):
    """Test hiding text message in audio file"""

    cover, path, hop_length = test_set_item
    print(f"Hop length: {hop_length}")

    # Calculate capacity
    _, H, W = cover.size()
    cover = cover[None].to(device)

    total_bits = H * W * data_depth
    print(f"Maximum capacity in bits: {total_bits}")

    payload = make_payload(W, H, data_depth, message)
    payload = payload.to(device)

    generated = test_audio(encoder, decoder, cover, payload, save_wav=True, hop_length=hop_length, file_path=path)

    text_return_ = make_message(generated, decoder=decoder, device=device)

    print('Message found: ', text_return_)


if __name__ == '__main__':
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_path = 'models/DenseEncoder_DenseDecoder_0.962_2025-04-05_22h46m42.dat'
    # model_path = 'models/DenseEncoder_DenseDecoder_0.964_2025-04-05_22h22m10.dat'
    # model_path = 'models/DenseEncoder_DenseDecoder_0.915_2025-04-09_12h08m20.dat'
    model_path = 'models\\gan_32_2_2_epochs_32\\DenseEncoder_DenseDecoder_0.966_2025-04-21_16h21m37.dat' #0.921
    # model_path = 'models\\gan_32_2_2_epochs_32\\DenseEncoder_DenseDecoder_0.961_2025-04-21_16h01m26.dat' #0.901
    # model_path = 'models\\gan_32_2_2_epochs_32\\DenseEncoder_DenseDecoder_0.956_2025-04-21_15h49m44.dat' #0.900 OK
    channels_size = 2
    data_depth = 2
    hidden_size = 32
    save_audio = False

    # Load models
    encoder, decoder = load_trained_models(
        model_path,
        data_depth=data_depth,
        hidden_size=hidden_size,
        channels_size=channels_size,
        device=device
    )

    # message = "a"
    message = "ka ka ka"
    # message = "This is secret message. It is very important to keep it secret"
    # message = "kasdjmaokasoddajio82isda9ajsd9asdkakalasdonma9a732jasc8ajnajsd9asdkasdkj aaaaaaaaaakasdjasjdjasd"

    data_dir="D:/Backup/FSDKaggle2018"
    # data_dir="C:/Users/Admin/Documents/GitHub/Steganography_GANs/audio/train"
    file_dir="D:\\Backup\\FSDKaggle2018\\test\\ed2e9480.wav"
    from torchvision import transforms
    transform = transforms.Compose([transforms.Lambda(lambda wav: audio_to_stft(wav))])
    # test_set = AudioToImageFolder(data_dir, transform=transform)
    # part_test_set = torch.utils.data.random_split(test_set, [100, len(test_set)-100])[0]
    # test_loader = torch.utils.data.DataLoader(part_test_set, batch_size=4, shuffle=True)
    test_single = SingleAudioLoader(file_dir, transform=transform)


    marked_spectrogram = test_hiding(encoder, decoder, test_single[0], message, data_depth, device)
