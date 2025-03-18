import numpy as np
import librosa
import matplotlib.pyplot as plt
from dataset import MusanNoiseDataset
import soundfile as sf
import os
from scipy.stats import pearsonr
from typing import Dict, Union

def get_audio_segment(audio_path: str, start_time: float, end_time: float, sr: int = 16000) -> np.ndarray:
    """Get specific segment from original audio file"""
    audio, _ = librosa.load(audio_path, sr=sr)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    return audio[start_sample:end_sample]


def mel_spectrogram_to_audio(mel_spectrogram, sr: int = 16000, n_iter=32):
    """
    Chuyển đổi Mel Spectrogram thành tín hiệu âm thanh.

    Args:
        mel_spectrogram (np.ndarray): Mel Spectrogram.
        sr (int): Tần số lấy mẫu.
        n_iter (int, optional): Số vòng lặp để khôi phục phase. Mặc định là 32.

    Returns:
        np.ndarray: Tín hiệu âm thanh khôi phục.
    """
    wav = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr, n_iter=n_iter)

    wav = np.clip(wav, -1.0, 1.0)  # Reduced clipping threshold

    return wav

def compare_signals(
    original: np.ndarray,
    noisy: np.ndarray,
    metadata: Dict,
    output_dir: str,
    sample_idx: int
) -> Dict[str, float]:
    """Compare original and reconstructed noisy signals"""
    os.makedirs(output_dir, exist_ok=True)

    # Ensure same length for comparison
    min_length = min(len(original), len(noisy))
    original = original[:min_length]
    noisy = noisy[:min_length]

    # Calculate metrics
    correlation, _ = pearsonr(original, noisy)
    mse = np.mean((original - noisy) ** 2)

    plt.figure(figsize=(15, 10))

    # Plot signals
    plt.subplot(2, 1, 1)
    plt.plot(original, 'b-', alpha=0.7)
    plt.title('Original Clean Audio')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(noisy, 'r-', alpha=0.7)
    plt.title('Reconstructed Noisy Audio')
    plt.grid(True)

    # Add metadata info
    segment_info = metadata['segment_info']
    plt.suptitle(
        f'File: {os.path.basename(metadata["clean_audio_path"])}\n' + \
        f'Segment: {segment_info["start_time"]:.1f}s - {segment_info["end_time"]:.1f}s\n' + \
        f'Correlation: {correlation:.3f}, MSE: {mse:.6f}'
    )
    plt.tight_layout()

    # Save results
    plt.savefig(os.path.join(output_dir, f'comparison_{sample_idx}.png'))
    plt.close()

    sf.write(os.path.join(output_dir, f'original_{sample_idx}.wav'), original, 16000)
    sf.write(os.path.join(output_dir, f'noisy_{sample_idx}.wav'), noisy, 16000)

    return {
        'correlation': correlation,
        'mse': mse
    }

if __name__ == "__main__":
    dataset = MusanNoiseDataset(dataset_dir="musan_dataset")
    output_dir = "reconstruction_results"

    for idx in range(0, 3):
        print(f"\nProcessing sample {idx}")

        # Get data
        data = dataset[idx]
        noisy_mel = data['noisy_mel'].numpy()
        metadata = data['metadata']

        # Get original audio segment
        original_audio = get_audio_segment(
            metadata['clean_audio_path'],
            metadata['segment_info']['start_time'],
            metadata['segment_info']['end_time']
        )

        # Reconstruct noisy audio
        reconstructed_noisy = mel_spectrogram_to_audio(noisy_mel)

        # Compare signals
        metrics = compare_signals(original_audio, reconstructed_noisy, metadata, output_dir, idx)

        print(f"Correlation: {metrics['correlation']:.3f}")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"Results saved in {output_dir}")
