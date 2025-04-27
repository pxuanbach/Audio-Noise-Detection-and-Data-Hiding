import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import librosa.display

def visualize_audio_comparison(original_path, modified_path, title="Audio Comparison"):
    """
    Visualize and compare two audio files
    :param original_path: Path to first audio file
    :param modified_path: Path to second audio file
    :param title: Title for the plot
    """
    # Read audio files
    sr1, audio1 = wavfile.read(original_path)
    sr2, audio2 = wavfile.read(modified_path)

    # Normalize audio data
    audio1 = audio1 / np.max(np.abs(audio1))
    audio2 = audio2 / np.max(np.abs(audio2))

    # Create time axis
    time1 = np.arange(len(audio1)) / sr1
    time2 = np.arange(len(audio2)) / sr2

    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot waveforms
    plt.subplot(3, 2, 1)
    plt.plot(time1, audio1)
    plt.title('Original Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 2, 2)
    plt.plot(time2, audio2)
    plt.title('Modified Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Calculate and plot spectrum
    freqs1 = np.fft.fftfreq(len(audio1), 1/sr1)
    spectrum1 = np.fft.fft(audio1)
    freqs2 = np.fft.fftfreq(len(audio2), 1/sr2)
    spectrum2 = np.fft.fft(audio2)

    plt.subplot(3, 2, 3)
    plt.plot(freqs1[:len(freqs1)//2], np.abs(spectrum1)[:len(spectrum1)//2])
    plt.title('Original Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    plt.subplot(3, 2, 4)
    plt.plot(freqs2[:len(freqs2)//2], np.abs(spectrum2)[:len(spectrum2)//2])
    plt.title('Modified Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    # Calculate and plot spectrogram using librosa
    D1 = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio1.astype(float))),
        ref=np.max
    )
    D2 = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio2.astype(float))),
        ref=np.max
    )

    plt.subplot(3, 2, 5)
    img1 = librosa.display.specshow(D1, sr=sr1, x_axis='time', y_axis='hz')
    plt.title('Original Spectrogram')
    plt.colorbar(img1, format='%+2.0f dB')

    plt.subplot(3, 2, 6)
    img2 = librosa.display.specshow(D2, sr=sr2, x_axis='time', y_axis='hz')
    plt.title('Modified Spectrogram')
    plt.colorbar(img2, format='%+2.0f dB')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("scripts/traditional_algo/visualization.png")

if __name__ == "__main__":
    # Example usage
    original_audio = "D:/Backup/musan/music/fma-western-art/music-fma-wa-0000.wav"
    modified_audio = "scripts/traditional_algo/stego_lsb.wav"
    visualize_audio_comparison(original_audio, modified_audio, "Original vs Stego Audio Comparison")
