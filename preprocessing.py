import os
import numpy as np
import librosa
import soundfile as sf
import random
from tqdm import tqdm
import json

# Path to MUSAN directory
MUSAN_PATH = "D:/Backup/musan"
MAX_DURATION = 180  # Maximum duration in seconds (3 minutes)
SAMPLE_RATE = 16000  # Sample rate
CONTINUES_DURATION = 1  # Maximum continuous noise duration
MIN_INTERVAL = 0.8  # Minimum interval between noise regions
MAX_REGIONS = 10  # Maximum number of noise regions

def segment_audio(audio: np.ndarray, sr: int = 16000, max_duration: int = MAX_DURATION) -> list:
    """
    Segment audio into chunks if longer than max_duration

    Args:
        audio: Audio signal
        sr: Sample rate
        max_duration: Maximum duration in seconds

    Returns:
        List of tuples (audio_segment, start_time, end_time)
    """
    duration = len(audio) / sr
    if duration <= max_duration:
        return [(audio, 0, duration)]

    max_samples = max_duration * sr
    segments = []

    for start in range(0, len(audio), max_samples):
        end = min(start + max_samples, len(audio))
        segment = audio[start:end]
        start_time = start / sr
        end_time = end / sr
        segments.append((segment, start_time, end_time))

    return segments

# Function to calculate energy and label optimal regions
def calculate_energy_and_label(audio, frame_size=1024, hop_length=512, energy_threshold=0.75):
    # Split into frames
    frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_length).T
    # Calculate average energy per frame
    energy = np.mean(frames ** 2, axis=1)
    # Label optimal regions (energy > percentile 75)
    threshold = np.percentile(energy, 100 * energy_threshold)
    labels = (energy > threshold).astype(int)
    return frames, energy, labels

# Function to add noise to optimal regions
def add_noise_to_optimal_regions(
    audio: np.ndarray,
    noise: np.ndarray,
    labels: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_size: int = 1024,
    hop_length: int = 512,
    min_interval: float = 0.5,
    continuous_duration: float = 0.2,
    amplitude_scale: float = 0.2,
    max_regions: int = 15
) -> tuple[np.ndarray, list]:
    """Add noise by modulating amplitude at optimal regions with uniform distribution"""
    noisy_audio = np.copy(audio)

    # Convert time parameters to frames
    min_frames = int(min_interval * sr / hop_length)
    max_continuous_frames = int(continuous_duration * sr / hop_length)

    # Find optimal regions
    optimal_frames = np.where(labels == 1)[0]

    # Đảm bảo phân phối đều trên toàn bộ audio
    total_frames = len(labels)
    if len(optimal_frames) > 0:
        # Chia audio thành max_regions phần bằng nhau
        segment_length = total_frames // max_regions
        selected_frames = []
        last_frame = -min_frames  # Khởi tạo frame cuối cùng
        continuous_count = 0  # Đếm số frame liên tục

        # Duyệt qua từng phần
        for i in range(max_regions):
            segment_start = i * segment_length
            segment_end = min((i + 1) * segment_length, total_frames)

            # Tìm các frame tối ưu trong đoạn hiện tại
            segment_optimal = optimal_frames[
                (optimal_frames >= segment_start) &
                (optimal_frames < segment_end)
            ]

            if len(segment_optimal) > 0:
                for frame in segment_optimal:
                    # Kiểm tra khoảng cách tối thiểu
                    if frame - last_frame >= min_frames:
                        # Kiểm tra số frame liên tục
                        if continuous_count < max_continuous_frames:
                            selected_frames.append(frame)
                            last_frame = frame
                            continuous_count += 1
                        else:
                            # Reset đếm frame liên tục và bỏ qua một khoảng
                            continuous_count = 0
                            last_frame = frame + min_frames

        # Normalize noise
        noise = noise / np.max(np.abs(noise))

        # Add noise at selected frames
        for frame in selected_frames:
            start = frame * hop_length
            end = start + frame_size
            if end <= len(audio):
                window = np.hanning(frame_size)
                noise_segment = noise[frame % len(noise):frame % len(noise) + frame_size]
                if len(noise_segment) < frame_size:
                    noise_segment = np.pad(noise_segment, (0, frame_size - len(noise_segment)))

                # Modulate amplitude
                audio_segment = audio[start:end]
                modulated_segment = audio_segment * (1 + amplitude_scale * noise_segment * window)
                noisy_audio[start:end] = modulated_segment

    return noisy_audio, selected_frames


def audio_to_mel_spectrogram(audio: np.ndarray, sr: int = 16000, n_mels=128):
    """
    Chuyển đổi tệp âm thanh thành Mel Spectrogram.

    Args:
        audio_path (str): Đường dẫn đến tệp âm thanh.
        sr (int, optional): Tần số lấy mẫu (sample rate). Mặc định là 22050 Hz.
        n_mels (int, optional): Số lượng Mel bands. Mặc định là 128.

    Returns:
        np.ndarray: Mel Spectrogram.
        int: Tần số lấy mẫu thực tế.
    """
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)

    return mel_spectrogram

def calculate_hiding_capacity(labels, selected_frames, frame_size=1024):
    """
    Calculate data hiding capacity in noise using LSB

    Args:
        labels (np.ndarray): Labels of optimal regions (0/1)
        frame_size (int): Frame size

    Returns:
        dict: Capacity information
        - total_bits: Total bits that can be hidden
        - total_bytes: Total bytes that can be hidden
        - optimal_frames: Number of optimal frames
        - capacity_per_second: Bits that can be hidden per second (assuming sr=16000)
    """
    optimal_frames = np.sum(labels == 1)
    total_bits = optimal_frames * frame_size  # Each sample can hide 1 bit
    total_bytes = total_bits // 8

    # Calculate capacity over time
    # With sr=16000 and hop_length=512:
    # - Each frame is 1024 samples = 0.064s
    # - Frames are 512 samples apart = 0.032s
    duration = len(labels) * 512 / 16000  # Total audio duration (seconds)
    capacity_per_second = total_bits / duration

    metadata = {
        "total_bits": int(total_bits),  # Convert np.int32 to int
        "total_bytes": int(total_bytes),  # Convert np.int32 to int
        "optimal_frames": int(optimal_frames),  # Convert np.int32 to int
        "capacity_per_second": float(capacity_per_second),  # Convert np.float64 to float
        "noise_distribution": {
            "selected_frames": len(selected_frames),
            "total_frames": len(labels),
            "coverage_ratio": float(len(selected_frames)) / len(labels)
        }
    }
    return metadata

# Function to preprocess MUSAN
def preprocess_musan(output_dir="musan_dataset"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata = {}
    sample_idx = 0

    # Get file lists
    speech_files = []
    music_files = []
    noise_files = []

    for root, _, files in os.walk(MUSAN_PATH):
        for file in files:
            if file.endswith(".wav"):
                if "speech" in root:
                    speech_files.append(os.path.join(root, file))
                elif "music" in root:
                    music_files.append(os.path.join(root, file))
                elif "noise" in root:
                    noise_files.append(os.path.join(root, file))

    clean_files = speech_files + music_files
    print(f"Found {len(clean_files)} clean files and {len(noise_files)} noise files")
    random.shuffle(clean_files)

    # Process each clean audio file
    for clean_file in tqdm(clean_files, desc="Processing files"):
        # Load clean audio
        audio, sr = librosa.load(clean_file, sr=SAMPLE_RATE)

        # Segment audio if needed
        segments = segment_audio(audio)

        for seg_idx, (audio_segment, start_time, end_time) in enumerate(segments):
            # Kiểm tra độ dài tối thiểu
            min_samples = 2048 + 512 * (128 - 1)  # n_fft + hop_length * (n_mels - 1)
            if len(audio_segment) < min_samples:
                print(f"Skip segment {seg_idx} in {clean_file}: too short ({len(audio_segment)} < {min_samples} samples)")
                continue

            # Select random noise file
            noise_file = random.choice(noise_files)
            noise, _ = librosa.load(noise_file, sr=SAMPLE_RATE)
            noise = noise[:len(audio_segment)]

            # Chuẩn hóa audio trước khi xử lý
            audio_segment = audio_segment / np.max(np.abs(audio_segment))

            # Calculate energy and labels
            frames, energy, labels = calculate_energy_and_label(audio_segment)

            # Add noise to optimal regions with new parameters
            noisy_audio, selected_frames = add_noise_to_optimal_regions(
                audio_segment,
                noise,
                labels,
                continuous_duration=CONTINUES_DURATION,
                max_regions=MAX_REGIONS,
                min_interval=MIN_INTERVAL
            )

            # Convert to mel spectrograms
            clean_mel = audio_to_mel_spectrogram(audio_segment, sr=SAMPLE_RATE)
            noisy_mel = audio_to_mel_spectrogram(noisy_audio, sr=SAMPLE_RATE)

            # Calculate hiding capacity with selected frames
            hiding_capacity = calculate_hiding_capacity(labels, selected_frames)

            # Add metadata
            metadata[f"sample_{sample_idx}"] = {
                "clean_audio_path": clean_file,
                "segment_info": {
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "duration": float(end_time - start_time)
                },
                "noise_audio_path": noise_file,
                "num_optimal_regions": int(np.sum(labels == 1)),
                "total_frames": len(labels),
                "hiding_capacity": hiding_capacity
            }

            # Save data
            output_prefix = os.path.join(output_dir, f"sample_{sample_idx}")
            np.save(f"{output_prefix}_clean_mel.npy", clean_mel)
            np.save(f"{output_prefix}_noisy_mel.npy", noisy_mel)
            np.save(f"{output_prefix}_noise.npy", noise)
            np.save(f"{output_prefix}_labels.npy", labels)

            sample_idx += 1

    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset created at: {output_dir}")
    print(f"Total samples: {sample_idx}")

if __name__ == "__main__":
    preprocess_musan()
