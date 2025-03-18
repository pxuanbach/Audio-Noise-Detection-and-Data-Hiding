import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
import torch
import os

def visualize_preprocessed_data(clean_mel, noisy_mel, noise, labels, metadata, output_path):
    """
    Visualize và lưu dữ liệu tiền xử lý:
    - Mel spectrogram của clean và noisy audio với nhãn vùng tối ưu
    """
    clean_mel = librosa.power_to_db(clean_mel.numpy(), ref=np.max)
    noisy_mel = librosa.power_to_db(noisy_mel.numpy(), ref=np.max)

    plt.figure(figsize=(15, 10))

    # Thêm tiêu đề chung với metadata
    plt.suptitle(f"Clean: {os.path.basename(metadata['clean_audio_path'])}\n" + \
                f"Noise: {os.path.basename(metadata['noise_audio_path'])}\n" + \
                f"Optimal regions: {metadata['num_optimal_regions']}/{metadata['total_frames']}",
                fontsize=10, y=1.02)

    # 1. Mel spectrogram âm thanh sạch
    plt.subplot(2, 2, 1)
    librosa.display.specshow(
        clean_mel,
        y_axis='mel',
        x_axis='time',
        sr=16000,
        hop_length=512
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram - Clean Audio")

    # 2. Mel spectrogram âm thanh có nhiễu
    plt.subplot(2, 2, 2)
    librosa.display.specshow(
        noisy_mel,
        y_axis='mel',
        x_axis='time',
        sr=16000,
        hop_length=512
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram - Noisy Audio")

    # 3. Audio waveform với optimal regions
    plt.subplot(2, 2, 3)
    # Load audio segment từ metadata
    segment_info = metadata['segment_info']
    audio = librosa.load(
        metadata['clean_audio_path'],
        sr=16000,
        offset=segment_info['start_time'],
        duration=segment_info['duration']
    )[0]

    # Tạo time axis cho đoạn audio được chọn
    time_axis = np.linspace(
        segment_info['start_time'],
        segment_info['end_time'],
        len(audio)
    )

    # Vẽ dạng sóng
    plt.plot(time_axis, audio, 'b-', alpha=0.5, linewidth=0.5)

    # Đánh dấu các vùng optimal - điều chỉnh thời gian theo segment
    for i, label in enumerate(labels.numpy()):
        if label == 1:
            start = segment_info['start_time'] + i * 512 / 16000
            end = segment_info['start_time'] + (i * 512 + 1024) / 16000
            plt.axvspan(start, end, color='r', alpha=0.2)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Clean Audio Waveform with Optimal Regions")
    plt.grid(True, alpha=0.3)

    # 4. Energy distribution với optimal threshold
    plt.subplot(2, 2, 4)
    # Tính lại energy từ audio gốc
    frames = librosa.util.frame(audio, frame_length=1024, hop_length=512).T
    energy = np.mean(frames ** 2, axis=1)
    frame_indices = np.arange(len(energy))

    # Vẽ energy của từng frame
    plt.plot(frame_indices, energy, 'b-', alpha=0.7, label='Frame Energy')

    # Vẽ ngưỡng threshold
    threshold = np.percentile(energy, 75)  # 75% threshold như trong preprocessing
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Threshold (75th percentile)')

    plt.xlabel("Frame Index")
    plt.ylabel("Energy")
    plt.title("Frame Energy Distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def collate_fn(batch):
    """
    Hàm collate_fn cho DataLoader

    Args:
        batch (List[Dict]): List các dictionary chứa dữ liệu mẫu

    Returns:
        Dict: Dữ liệu đã được xử lý
    """
    # Khởi tạo lists rỗng cho từng loại dữ liệu
    clean_audios = []
    noisy_audios = []
    noises = []
    labels_list = []
    metadata_list = []

    # Gom dữ liệu từ các dictionary trong batch
    for sample in batch:
        clean_audios.append(sample["clean_mel"])
        noisy_audios.append(sample["noisy_mel"])
        noises.append(sample["noise"])
        labels_list.append(sample["labels"])
        metadata_list.append(sample["metadata"])

    # Stack các tensor
    return {
        "clean_mel": torch.stack(clean_audios),
        "noisy_mel": torch.stack(noisy_audios),
        "noise": torch.stack(noises),
        "labels": torch.stack(labels_list),
        "metadata": metadata_list
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import MusanNoiseDataset

    # Tạo thư mục để lưu visualization
    viz_dir = "visualization_results"
    os.makedirs(viz_dir, exist_ok=True)

    # Load dataset
    dataset = MusanNoiseDataset(dataset_dir="musan_dataset")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Saving visualizations to: {viz_dir}")

    # Visualize một số mẫu
    num_samples_to_show = 3

    for i, batch in enumerate(dataloader):
        if i >= num_samples_to_show:
            break

        print(f"\nProcessing sample {i+1}/{num_samples_to_show}")

        clean_mel = batch["clean_mel"][0]
        noisy_mel = batch["noisy_mel"][0]
        noise = batch["noise"][0]
        labels = batch["labels"][0]
        metadata = batch["metadata"][0]

        print(f"Clean mel shape: {clean_mel.shape}")
        print(f"Noisy mel shape: {noisy_mel.shape}")
        print(f"Noise shape: {noise.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Metadata: {metadata}")
        print("\nHiding Capacity Information:")
        hiding_capacity = metadata["hiding_capacity"]
        print(f"- Total bits: {hiding_capacity['total_bits']:,} bits")
        print(f"- Total bytes: {hiding_capacity['total_bytes']:,} bytes")
        print(f"- Capacity per second: {hiding_capacity['capacity_per_second']:.2f} bits/second")

        output_path = os.path.join(viz_dir, f"sample_{i+1}_visualization.png")
        visualize_preprocessed_data(clean_mel, noisy_mel, noise, labels, metadata, output_path)
        print(f"Saved visualization to: {output_path}")
