import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from models.unet_lstm import UNetLSTM
from datetime import datetime
import librosa
import soundfile as sf


class NoisePredictor:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNetLSTM().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Constants
        self.sr = 16000
        self.n_mels = 128
        self.hop_length = 512
        self.frame_size = 1024

        # Add supported formats
        self.supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Preprocess audio file to mel spectrogram"""
        # Check file format
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(self.supported_formats)}")

        try:
            # Try loading with soundfile first
            audio, _ = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert stereo to mono
        except Exception as e:
            print(f"SoundFile failed: {str(e)}")
            print("Trying librosa backend...")
            try:
                audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load audio file. Error: {str(e)}\n"
                    "Please ensure you have the required audio codecs installed:\n"
                    "1. Install ffmpeg: https://ffmpeg.org/download.html\n"
                    "2. Add ffmpeg to system PATH\n"
                    "3. Install additional codecs if needed"
                )

        # Normalize audio
        audio = audio / np.max(np.abs(audio))

        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )

        # Create 3-channel input (original, copied, zero)
        mel_input = np.stack([mel_spec, mel_spec, np.zeros_like(mel_spec)])
        return torch.FloatTensor(mel_input)

    def predict(self, audio_path: str):
        """Predict noise regions in audio"""
        # Preprocess audio
        mel_input = self.preprocess_audio(audio_path)
        mel_input = mel_input.unsqueeze(0).to(self.device)  # Add batch dimension

        # Get predictions
        with torch.no_grad():
            predictions = self.model(mel_input)
            predictions = (predictions > 0.5).float().cpu().numpy()[0, 0]

        return predictions

    def visualize_predictions(self, audio_path: str, output_dir: str = None):
        """Visualize audio and predictions"""
        # Create output directory
        if output_dir is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("predictions", date_str)
        os.makedirs(output_dir, exist_ok=True)

        # Load audio and get predictions
        audio, _ = librosa.load(audio_path, sr=self.sr)
        audio_duration = len(audio) / self.sr  # Duration in seconds
        predictions = self.predict(audio_path)

        # Calculate number of prediction frames needed
        num_frames = int(audio_duration * self.sr / self.hop_length)
        predictions = predictions[:num_frames]  # Truncate predictions to match audio length

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

        # Plot waveform with correct time axis
        time_axis = np.arange(len(audio)) / self.sr
        ax1.plot(time_axis, audio, 'b-', alpha=0.7)
        ax1.set_title('Audio Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_xlim([0, audio_duration])  # Set x-axis limits to audio duration

        # Highlight predicted noise regions with correct timing
        pred_time = np.linspace(0, audio_duration, len(predictions))
        for i, pred in enumerate(predictions):
            if pred > 0.5:
                start = pred_time[i]
                end = min(start + self.frame_size / self.sr, audio_duration)
                ax1.axvspan(start, end, color='r', alpha=0.2)

        # Plot predictions with matching time axis
        ax2.plot(pred_time, predictions, 'r-', label='Noise Probability')
        ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        ax2.set_title('Noise Predictions')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Probability')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim([0, audio_duration])  # Set x-axis limits to audio duration

        # Save results
        plot_path = os.path.join(output_dir, f'prediction_{os.path.basename(audio_path)}.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        # Add duration info to return values
        return predictions, plot_path, audio_duration

def main():
    # Parameters
    MODEL_PATH = "models/best_model.pth"

    # Initialize predictor
    try:
        predictor = NoisePredictor(MODEL_PATH)
    except Exception as e:
        print(f"Error initializing predictor: {str(e)}")
        return

    # Example usage
    while True:
        audio_path = input("\nEnter path to audio file (or 'q' to quit): ")

        if audio_path.lower() == 'q':
            break

        if not os.path.exists(audio_path):
            print(f"Error: File {audio_path} not found")
            continue

        print("Processing audio...")
        predictions, plot_path, duration = predictor.visualize_predictions(audio_path)
        print(f"\nPredictions saved to: {plot_path}")
        print(f"Audio duration: {duration:.2f} seconds")
        print(f"Found {np.sum(predictions > 0.5)} potential noise regions")


if __name__ == "__main__":
    main()
