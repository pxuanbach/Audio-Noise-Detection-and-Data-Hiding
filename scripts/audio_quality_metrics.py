import numpy as np
from scipy.io import wavfile
import soundfile as sf
import os

def calculate_snr(original, stego):
    """Calculate Signal-to-Noise Ratio between original and stego audio"""
    noise = original - stego
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_rcc(original, stego, window_size=1024):
    """Calculate average correlation coefficient between original and stego audio using windowed segments"""
    num_segments = len(original) // window_size
    correlations = []

    for i in range(num_segments):
        start = i * window_size
        end = start + window_size
        orig_segment = original[start:end]
        stego_segment = stego[start:end]

        # Calculate correlation for this segment
        correlation = np.corrcoef(orig_segment, stego_segment)[0,1]
        correlations.append(correlation)

    # Return average correlation across all segments
    return np.mean(correlations)

def get_file_size_kb(file_path):
    """Get file size in kilobytes"""
    return os.path.getsize(file_path) / 1024

def analyze_audio_quality(original_path, stego_path):
    """Analyze audio quality metrics between original and stego audio"""
    # Read audio files
    orig_sr, original = wavfile.read(original_path)
    stego_sr, stego = wavfile.read(stego_path)

    # Get file sizes
    stego_size = get_file_size_kb(stego_path)

    # Convert to float32 if needed
    if original.dtype != np.float32:
        original = original.astype(np.float32) / np.iinfo(original.dtype).max
    if stego.dtype != np.float32:
        stego = stego.astype(np.float32) / np.iinfo(stego.dtype).max

    # Calculate metrics
    snr_value = calculate_snr(original, stego)
    rcc_value = calculate_rcc(original, stego)

    return snr_value, rcc_value, stego_size

if __name__ == "__main__":
    # Example usage
    # original_path = "D:/Backup/FSDKaggle2018/test/0bb807c0.wav"
    # original_path = "D:/Backup/FSDKaggle2018/test/6e32975d.wav"
    # original_path = "D:/Backup/FSDKaggle2018/test/0db65bf4.wav"
    original_path = "D:/Backup/FSDKaggle2018/test/008afd93.wav"
    # stego_path = "scripts/traditional_algo/stego_dct.wav"
    # stego_path = "scripts/traditional_algo/stego_lsb.wav"
    stego_path = "scripts/traditional_algo/stego_dct.wav"

    try:
        snr, rcc, stego_size = analyze_audio_quality(original_path, stego_path)
        print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")
        print(f"Correlation Coefficient (RCC): {rcc:.4f}")
        print(f"Stego File Size: {stego_size:.2f} KB")
    except Exception as e:
        print(f"Error: {str(e)}")
