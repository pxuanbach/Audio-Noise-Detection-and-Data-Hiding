import librosa
import numpy as np
import skimage
import torch


def compute_hop_length(segment_len, n_fft, target_frames):
    return (segment_len - n_fft) // (target_frames - 1)


def audio_to_stft(audio_wav, target_frames=360):
    audio_wav = audio_wav[0]
    audio_length = audio_wav.shape[0]
    hop_length = int(audio_length / (target_frames - 1))
    n_fft = int((target_frames - 1) * 2)

    mel_spectrogram = librosa.stft(
        np.asanyarray(audio_wav),
        n_fft=n_fft,
        hop_length=hop_length
    )

    mel_real = np.real(mel_spectrogram)
    mel_imag = np.imag(mel_spectrogram)

    if mel_real.shape != (target_frames, target_frames):
        mel_real = skimage.transform.resize(
            image=mel_real,
            output_shape=(target_frames, target_frames),
            order=1
        )

    if mel_imag.shape != (target_frames, target_frames):
        mel_imag = skimage.transform.resize(
            image=mel_imag,
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
