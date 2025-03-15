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


def visualize_spectrograms(noisy_mel, mask, sr=16000, hop_length=256, output_path=None):
    """
    Visualize mel spectrogram and noise mask. Save to file if output_path is provided, otherwise display.

    Args:
        noisy_mel: Mel spectrogram
        mask: Binary noise mask
        sr: Sample rate
        hop_length: Hop length for spectrogram
        output_path: Optional path to save the output image. If None, display instead.
    """
    # Convert tensors to numpy and remove batch dimension
    if isinstance(noisy_mel, torch.Tensor):
        noisy_mel = noisy_mel.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    noisy_mel = np.squeeze(noisy_mel)
    mask = np.squeeze(mask)

    # Convert mel to dB scale
    mel_db = librosa.power_to_db(noisy_mel, ref=np.max)

    # Calculate time and frequency axes
    times = np.arange(mask.shape[1]) * hop_length / sr
    freqs = np.arange(mask.shape[0])

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

    # Plot original mel spectrogram
    img1 = librosa.display.specshow(
        mel_db,
        y_axis='mel',
        x_axis='time',
        sr=sr,
        hop_length=hop_length,
        ax=ax1,
        cmap='viridis'
    )
    ax1.set_title('Noisy Mel Spectrogram')
    fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

    # Plot binary mask with clear colors
    img2 = ax2.imshow(
        mask,
        aspect='auto',
        origin='lower',
        cmap='RdYlBu_r',  # Red for noise (1), Blue for clean (0)
        vmin=0,
        vmax=1,
        extent=[times[0], times[-1], freqs[0], freqs[-1]]
    )
    ax2.set_title('Noise Mask (Red: Noise, Blue: Clean)')
    fig.colorbar(img2, ax=ax2)

    # Plot overlay of mask on mel spectrogram
    masked_mel = mel_db.copy()
    masked_mel[mask > 0.5] = np.min(mel_db)  # Highlight noisy regions
    img3 = librosa.display.specshow(
        masked_mel,
        y_axis='mel',
        x_axis='time',
        sr=sr,
        hop_length=hop_length,
        ax=ax3,
        cmap='viridis'
    )
    ax3.set_title('Mel Spectrogram with Noise Regions Highlighted')
    fig.colorbar(img3, ax=ax3, format='%+2.0f dB')

    # Add time markers and mel frequency labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mel Frequency')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Spectrograms saved to: {output_path}")
    else:
        plt.show()

    noise_percentage = (mask > 0.5).mean() * 100
    print(f"\nNoise Statistics:")
    print(f"Percentage of regions with noise: {noise_percentage:.1f}%")


