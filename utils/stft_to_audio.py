import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

def stft_to_audio(stft_data, sr=22000, n_fft=719, max_duration=10.0):
    """Convert single STFT data back to audio using inverse STFT"""
    # Get real and imaginary parts
    real_part = stft_data[0].astype(np.float64)  # Convert to float64 for stability
    imag_part = stft_data[1].astype(np.float64)  # Convert to float64 for stability

    # Verify data is not corrupted (should not be all zeros)
    if np.all(real_part == 0) and np.all(imag_part == 0):
        raise ValueError("STFT data appears to be corrupted (all zeros)")

    # Reconstruct complex STFT
    complex_stft = real_part + 1j * imag_part

    # Estimate required audio length (in samples) for 10 seconds at sr=22000
    expected_samples = max_duration * sr

    # Calculate hop_length to achieve correct duration
    hop_length = int(expected_samples // (complex_stft.shape[1] - 1))  # Ensure integer

    # Verify hop_length is valid
    if hop_length < 1:
        hop_length = 1
        print("Warning: hop_length was adjusted to minimum value of 1")

    # Apply ISTFT with correct parameters
    try:
        audio = librosa.istft(
            complex_stft,
            hop_length=hop_length,
            win_length=n_fft,
            n_fft=n_fft,
            center=True,  # Add center padding
            length=int(expected_samples)  # Ensure integer
        )
    except Exception as e:
        print(f"ISTFT failed with error: {str(e)}")
        print(f"Shape: {complex_stft.shape}, hop_length: {hop_length}")
        raise

    # Check if audio is too quiet or contains NaN values
    if np.any(np.isnan(audio)) or np.abs(audio).max() < 1e-3:
        # Normalize if too quiet or contains NaN
        audio = librosa.util.normalize(np.nan_to_num(audio))

    return audio, sr

def convert_file(input_path, output_path, sr=22000):
    """Convert single .npy file to .wav"""
    # Load STFT data
    stft_data = np.load(input_path)

    # Convert to audio
    audio, sr = stft_to_audio(stft_data)

    # Save audio file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr)

if __name__ == "__main__":
    # Example usage for single file
    input_file = "datasets/processed/test/music/music-fma-0001.npy"
    output_file = "datasets/reconstructed/music-fma-0001.wav"
    convert_file(input_file, output_file)
    print(f"Audio file saved to {output_file}")
