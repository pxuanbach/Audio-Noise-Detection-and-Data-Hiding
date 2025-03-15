import torch
import librosa
import numpy as np
from train import CNN_LSTM
import os

class NoiseDetector:
    def __init__(self, model_path, sr=16000, n_mels=128, max_frames=186):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN_LSTM().to(self.device)

        # Load trained model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Audio processing parameters
        self.sr = sr
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.n_fft = 1024
        self.hop_length = 256
        self.segment_duration = 0.5  # 500ms segments

    def process_audio(self, audio_path):
        """
        Process audio file and return time segments containing noise.

        Args:
            audio_path (str): Path to audio file

        Returns:
            list: List of tuples containing (start_time, end_time) of noisy segments
            numpy.ndarray: Boolean array indicating noisy frames
        """
        # Load and process audio
        audio, _ = librosa.load(audio_path, sr=self.sr)

        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)

        # Process in windows if audio is too long
        predictions = []
        for i in range(0, mel_spec.shape[1], self.max_frames):
            # Extract window
            window = mel_spec[:, i:i + self.max_frames]
            if window.shape[1] < self.max_frames:
                window = np.pad(window, ((0, 0), (0, self.max_frames - window.shape[1])), mode='constant')

            # Convert to tensor
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            window_tensor = window_tensor.to(self.device)

            # Get model predictions
            with torch.no_grad():
                output = self.model(window_tensor)
                pred = (output > 0.5).float().cpu().numpy().squeeze()

            # Only keep predictions for actual frames (not padding)
            actual_frames = min(self.max_frames, mel_spec.shape[1] - i)
            predictions.extend(pred[0, :actual_frames])

        # Convert frame-level predictions to time segments
        noisy_segments = []
        frame_duration = self.hop_length / self.sr
        is_noisy = False
        start_time = 0

        predictions = np.array(predictions)
        frame_indicators = predictions > 0.5

        for i, is_noise in enumerate(frame_indicators):
            if is_noise and not is_noisy:
                start_time = i * frame_duration
                is_noisy = True
            elif not is_noise and is_noisy:
                end_time = i * frame_duration
                noisy_segments.append((start_time, end_time))
                is_noisy = False

        # Handle case where audio ends during noisy segment
        if is_noisy:
            end_time = len(frame_indicators) * frame_duration
            noisy_segments.append((start_time, end_time))

        return noisy_segments, frame_indicators

def main():
    # Example usage
    detector = NoiseDetector(
        model_path="models/250316_cnn_lstm_loss_0.113_acc_0.9383.pth"
    )

    audio_path = "output.wav"
    noisy_segments, frame_indicators = detector.process_audio(audio_path)

    print("\nNoisy segments detected:")
    for start, end in noisy_segments:
        print(f"From {start:.2f}s to {end:.2f}s")

    print(f"\nTotal frames: {len(frame_indicators)}")
    print(f"Noisy frames: {np.sum(frame_indicators)}")

if __name__ == "__main__":
    main()
