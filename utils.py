import librosa
import librosa.display
import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt


def mel_to_audio(mel_spec, sr=16000, n_fft=1024, hop_length=256):
    """Convert mel spectrogram from dataset back to audio with improved stability"""
    try:
        # Convert tensor to numpy and remove batch dimension
        if isinstance(mel_spec, torch.Tensor):
            mel_spec = mel_spec.detach().cpu().numpy()
        mel_spec = np.squeeze(mel_spec)

        # Apply log compression and proper scaling
        mel_spec = np.clip(mel_spec, 1e-10, None)  # Avoid too small values
        mel_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
        mel_db = np.clip(mel_db, -80.0, 0.0)  # Clip to reasonable dB range

        # Convert back to power spectrum with controlled scaling
        mel_power = librosa.db_to_power(mel_db, ref=1.0)
        mel_power = np.clip(mel_power, 0, 1.0)  # Ensure valid power range

        # Convert to linear spectrogram
        linear_spec = librosa.feature.inverse.mel_to_stft(
            mel_power,
            sr=sr,
            n_fft=n_fft,
            power=1.0  # Changed to 1.0 since we already have power spectrum
        )

        # Additional safety check
        if not np.all(np.isfinite(linear_spec)):
            print("Warning: Linear spectrogram contains non-finite values")
            linear_spec = np.nan_to_num(linear_spec, nan=0.0, posinf=1.0, neginf=0.0)

        # Reconstruct audio with more iterations for better quality
        y = librosa.griffinlim(
            linear_spec,
            n_iter=64,  # Increased iterations
            hop_length=hop_length,
            win_length=n_fft,
            window='hann',
            center=True,
            momentum=0.99,
            random_state=0  # For reproducibility
        )

        # Final normalization with safety checks
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print("Warning: Audio contains non-finite values after Griffin-Lim")
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        return y

    except Exception as e:
        print(f"Error converting mel to audio: {str(e)}")
        return np.zeros(sr)


def visualize_spectrograms(noisy_mel, mask, sr=16000, hop_length=256, output_path=None, ax=None, title=None):
    """
    Visualize mel spectrogram and noise mask.
    """
    # Convert tensors to numpy and remove batch dimension
    if isinstance(noisy_mel, torch.Tensor):
        noisy_mel = noisy_mel.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Ensure we have 2D arrays
    if noisy_mel.ndim == 3:
        noisy_mel = np.squeeze(noisy_mel, axis=0)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=0)

    # Calculate time and frequency axes
    times = np.arange(noisy_mel.shape[1]) * hop_length / sr
    freqs = np.arange(noisy_mel.shape[0])

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    else:
        ax1, ax2 = ax

    # Plot noisy mel spectrogram
    img1 = ax1.imshow(
        noisy_mel,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[times[0], times[-1], freqs[0], freqs[-1]]
    )
    ax1.set_title('Noisy Mel Spectrogram' if title is None else f'{title} - Mel')
    ax1.set_ylabel('Mel Frequency')
    plt.colorbar(img1, ax=ax1)

    # Plot mask
    img2 = ax2.imshow(
        mask,
        aspect='auto',
        origin='lower',
        cmap='RdYlBu_r',
        vmin=0,
        vmax=1,
        extent=[times[0], times[-1], freqs[0], freqs[-1]]
    )
    ax2.set_title('Noise Mask' if title is None else f'{title} - Mask')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Mel Frequency')
    plt.colorbar(img2, ax=ax2)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    return (ax1, ax2)


