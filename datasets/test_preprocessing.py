import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed")
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")  # Added train directory path

def plot_stft_channels(stft_data, title, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Convert to dB scale for better visualization
    real_db = librosa.amplitude_to_db(np.abs(stft_data[0]), ref=np.max)
    imag_db = librosa.amplitude_to_db(np.abs(stft_data[1]), ref=np.max)

    # Plot real part
    img1 = librosa.display.specshow(real_db,
                                  y_axis='linear',
                                  x_axis='time',
                                  ax=ax1,
                                  cmap='magma')
    ax1.set_title('Real Part')
    fig.colorbar(img1, ax=ax1, format="%+2.f dB")

    # Plot imaginary part
    img2 = librosa.display.specshow(imag_db,
                                  y_axis='linear',
                                  x_axis='time',
                                  ax=ax2,
                                  cmap='magma')
    ax2.set_title('Imaginary Part')
    fig.colorbar(img2, ax=ax2, format="%+2.f dB")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def test_preprocessing_output():
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(PROCESSED_DIR), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    categories = ['speech', 'music']

    for category in categories:
        category_dir = os.path.join(TRAIN_DIR, category)  # Changed to use train directory
        if not os.path.exists(category_dir):
            print(f"No processed files found for {category} in train set")
            continue

        # Get first 3 samples from each category in train set
        sample_files = [f for f in os.listdir(category_dir) if f.endswith('.npy')][:3]

        for sample in sample_files:
            stft_data = np.load(os.path.join(category_dir, sample))
            title = f"{category.capitalize()} Train Sample: {sample}"  # Updated title
            save_path = os.path.join(plots_dir, f"train_{os.path.splitext(sample)[0]}_visualization.png")
            plot_stft_channels(stft_data, title, save_path)
            print(f"Saved visualization for train sample {sample}")

if __name__ == "__main__":
    test_preprocessing_output()
    print(f"Train sample visualizations saved to {os.path.join(os.path.dirname(PROCESSED_DIR), 'plots')}")
