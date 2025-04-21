import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize_audio(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path)

    # Create figure with 3 subplots
    plt.figure(figsize=(15, 10))

    # Plot waveform (time domain)
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Time Domain Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(D), ref=np.min)
    librosa.display.specshow(db, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    # Plot mel spectrogram
    plt.subplot(3, 1, 3)
    sgram, _ = librosa.magphase(D)
    mel_spect = librosa.feature.melspectrogram(S=sgram, sr=sr)
    mel_db = librosa.amplitude_to_db(mel_spect, ref=np.min)
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace with your audio file path
    audio_file = "D:\\Backup\\musan\\speech\\librivox\\speech-librivox-0000.wav"
    visualize_audio(audio_file)
