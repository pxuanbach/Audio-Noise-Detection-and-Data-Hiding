import librosa
import numpy as np
import skimage
import soundfile as sf
import torch
import skimage.transform
import torchaudio

from .chaotic_sequence import get_segment_positions


def compute_hop_length(segment_len, n_fft, target_frames):
    return (segment_len - n_fft) // (target_frames - 1)


# def audio_to_stft(audio_path, sr=22000, n_fft=719, segment_len_sec=10, target_frames=360, target_freq_bins=360,
#                  max_segments=None, chaotic_key=0.1, augment=False, aug_params=None, dtype=np.float32):
#     """Convert audio file to STFT segments with optional augmentation

#     Args:
#         audio_path: Path to the audio file
#         sr: Sampling rate (default changed to 22000)
#         n_fft: Number of FFT components (719 for 360 frequency bins)
#         segment_len_sec: Segment length in seconds
#         target_frames: Target number of time frames (default 360)
#         target_freq_bins: Target number of frequency bins (default 360)
#         max_segments: Maximum number of segments to return
#         chaotic_key: Key for chaotic sequence generation
#         augment: Whether to apply augmentation
#         aug_params: Parameters for augmentation
#         dtype: Data type for output arrays
#     """
#     # Load and normalize audio
#     y, sr = librosa.load(audio_path, sr=sr)

#     if augment and aug_params:
#         # Time stretching
#         if np.random.random() < 0.5:
#             stretch_factor = np.random.uniform(*aug_params['time_stretch'])
#             y = librosa.effects.time_stretch(y, rate=stretch_factor)

#         # Pitch shifting
#         if np.random.random() < 0.5:
#             n_steps = np.random.uniform(*aug_params['pitch_shift'])
#             y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

#         # Add noise
#         if np.random.random() < 0.5:
#             noise_level = np.random.uniform(*aug_params['noise_level'])
#             noise = np.random.normal(0, noise_level, len(y))
#             y = y + noise

#     # Calculate segment length in samples
#     segment_len = int(segment_len_sec * sr)

#     segments = []
#     positions = get_segment_positions(
#         audio_len=len(y),
#         segment_len=segment_len,
#         max_segments=max_segments,
#         key=chaotic_key
#     )

#     for pos in positions:
#         segment = y[pos:pos + segment_len]

#         if len(segment) < segment_len:
#             continue

#         # Compute hop length if not provided
#         hop_length = compute_hop_length(len(segment), n_fft, target_frames)

#         # Compute STFT
#         stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)

#         # Convert to real/imaginary and set dtype
#         real_part = np.real(stft).astype(dtype)
#         imag_part = np.imag(stft).astype(dtype)

#         # Fix dimensions to target size
#         real_part = librosa.util.fix_length(real_part, size=target_freq_bins, axis=0)
#         real_part = librosa.util.fix_length(real_part, size=target_frames, axis=1)
#         imag_part = librosa.util.fix_length(imag_part, size=target_freq_bins, axis=0)
#         imag_part = librosa.util.fix_length(imag_part, size=target_frames, axis=1)

#         features = np.stack([real_part, imag_part], axis=0)
#         segments.append(features)

#     return segments, sr, n_fft, hop_length


def audio_to_stft(audio_wav, target_frames=360):
    if audio_wav.dim() > 1:
        audio_wav = audio_wav[0]
    
    audio_tensor = audio_wav.to(dtype=torch.float32)
    device = audio_tensor.device

    audio_length = audio_tensor.shape[0]
    hop_length = int(audio_length / (target_frames - 1))
    n_fft = int((target_frames - 1) * 2)

    # Use Hann window to match librosa.stft
    window = torch.hann_window(n_fft, device=device)

    # Pad the signal to match librosa's center=True
    stft_transform = torch.stft(
        audio_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,  # Match librosa default
        pad_mode='reflect',  # Match librosa padding
        return_complex=True
    )

    # Normalize to match librosa (approximate scaling)
    # librosa divides by sqrt(sum(window^2)) for energy normalization
    window_sum = torch.sqrt(torch.sum(window ** 2))
    stft_transform = stft_transform / window_sum

    mel_real = stft_transform.real
    mel_imag = stft_transform.imag

    if mel_real.shape != (target_frames, target_frames):
        mel_real_np = mel_real.cpu().numpy()
        mel_real_np = skimage.transform.resize(
            mel_real_np,
            output_shape=(target_frames, target_frames),
            order=1
        )
        mel_real = torch.tensor(mel_real_np, dtype=torch.float32, device=device)

    if mel_imag.shape != (target_frames, target_frames):
        mel_imag_np = mel_imag.cpu().numpy()
        mel_imag_np = skimage.transform.resize(
            mel_imag_np,
            output_shape=(target_frames, target_frames),
            order=1
        )
        mel_imag = torch.tensor(mel_imag_np, dtype=torch.float32, device=device)

    output = torch.stack([mel_real, mel_imag])

    return output, hop_length, n_fft

def audio_to_stft_alt(audio_wav, target_frames=360):
    audio_wav = audio_wav[0]
    audio_tensor = torch.tensor(audio_wav, dtype=torch.float32)

    audio_length = audio_tensor.shape[0]
    hop_length = int(audio_length / (target_frames - 1))
    n_fft = int((target_frames - 1) * 2)

    # Thực hiện STFT bằng torchaudio
    stft_result = torch.stft(
        audio_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )

    # Chuyển sang numpy để resize bằng skimage
    stft_numpy = stft_result.cpu().detach().numpy()

    # Lấy real và imag
    mel_real = np.real(stft_numpy)
    mel_imag = np.imag(stft_numpy)

    # Resize nếu cần
    if mel_real.shape != (target_frames, target_frames):
        mel_real = skimage.transform.resize(
            mel_real,
            output_shape=(target_frames, target_frames),
            order=1
        )

    if mel_imag.shape != (target_frames, target_frames):
        mel_imag = skimage.transform.resize(
            mel_imag,
            output_shape=(target_frames, target_frames),
            order=1
        )

    return torch.tensor([mel_real, mel_imag]).float(), hop_length, n_fft



def stft_to_audio(stft_array, hop_length):
    """Convert STFT data back to audio

    Args:
        stft_array: STFT data array [2, freq, time] containing real and imaginary parts
        sr: Sampling rate
        n_fft: Number of FFT components
        hop_length: Hop length for STFT

    Returns:
        audio: Reconstructed audio signal
    """
    real_part = stft_array[0]
    imag_part = stft_array[1]
    complex_stft = real_part + 1j * imag_part
    audio = librosa.istft(complex_stft, hop_length=hop_length)
    return audio
