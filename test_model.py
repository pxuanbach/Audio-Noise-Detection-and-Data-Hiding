import torch
from inference import load_trained_models, test_hiding, make_message
from datasets import SingleAudioLoader
from utils.audio_to_stft import audio_to_stft
from torchvision import transforms
import os
from tabulate import tabulate

CHANNELS_SIZE = 2
DATA_DEPTH = 2
HIDDEN_SIZE = 32
TARGET_FRAMES = 360

# Test messages
TEST_MESSAGES = [
    "a",
    "ka ka ka",
    "This is secret message. It is very important to keep it secret",
    "kasdjmaokasoddajio82isda9ajsd9asdkakalasdonma9a732jasc8ajnajsd9asdkasdkj aaaaaaaaaakasdjasjdjasd",
    "It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English."
]

# Test audio files
FSD_KAGGLE_TEST_DIR = "D:/Backup/FSDKaggle2018/test/"
AUDIO_FILES = [
    f"{FSD_KAGGLE_TEST_DIR}/ed2e9480.wav",    # Clarinet
    f"{FSD_KAGGLE_TEST_DIR}/0fe4e425.wav",    # Oboe
    f"{FSD_KAGGLE_TEST_DIR}/1e446435.wav",    # Flute
    f"{FSD_KAGGLE_TEST_DIR}/4c7201d9.wav",    # Cello
    f"{FSD_KAGGLE_TEST_DIR}/5d8faaeb.wav",    # Clarinet
    f"{FSD_KAGGLE_TEST_DIR}/6ba93bc4.wav",    # Applause
    f"{FSD_KAGGLE_TEST_DIR}/7c56f810.wav",    # Saxophone
    f"{FSD_KAGGLE_TEST_DIR}/0a5cbf90.wav",    # Glockenspiel
    f"{FSD_KAGGLE_TEST_DIR}/f4e4574e.wav",    # Acoustic_guitar
    f"{FSD_KAGGLE_TEST_DIR}/1c01994c.wav",    # Computer_keyboard
]

def test_model(model_path: str):
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Load models
    encoder, decoder = load_trained_models(
        model_path,
        data_depth=DATA_DEPTH,
        hidden_size=HIDDEN_SIZE,
        channels_size=CHANNELS_SIZE,
        device=device
    )

    results = {}

    transform = transforms.Compose([
        transforms.Lambda(lambda wav: audio_to_stft(wav, TARGET_FRAMES))
    ])

    # Test each combination
    for audio_file in AUDIO_FILES:
        filename = os.path.basename(audio_file)
        results[filename] = {'total': len(TEST_MESSAGES), 'success': 0, 'failures': []}

        print(f"\nTesting audio file: {filename}")

        test_single = SingleAudioLoader(audio_file, transform=transform)

        for idx, message in enumerate(TEST_MESSAGES):
            try:
                extracted_message = test_hiding(encoder, decoder, test_single[0], message, DATA_DEPTH, device)

                if extracted_message == message:
                    results[filename]['success'] += 1
                else:
                    results[filename]['failures'].append(str(idx))

            except Exception as e:
                print(f"Test failed: {str(e)}")
                results[filename]['failures'].append(str(idx))

    print("\nTest Results Summary:", model_path)
    print("=" * 80)

    table_data = []
    for filename, result in results.items():
        success_rate = (result['success'] / result['total']) * 100
        table_data.append([
            filename,
            f"{result['success']}/{result['total']}",
            f"{success_rate:.1f}%",
            ', '.join(result['failures']) if result['failures'] else 'None'
        ])

    headers = ["Audio File", "Success/Total", "Success Rate", "Failed Messages (Indices)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    model_path = 'models\\lstm_gan_32_2_2_epochs_32\\ImprovedEncoder_ImprovedDecoder_0.955_2025-05-14_15h26m35.dat'
    # model_path = 'models\\lstm_gan_32_2_2_epochs_32\\ImprovedEncoder_ImprovedDecoder_0.949_2025-05-14_16h37m57.dat'
    # model_path = 'models\\lstm_gan_32_2_2_epochs_32\\ImprovedEncoder_ImprovedDecoder_0.931_2025-05-14_16h44m11.dat'
    # model_path = 'models\\lstm_gan_32_2_2_epochs_32\\ImprovedEncoder_ImprovedDecoder_0.925_2025-05-14_16h59m49.dat'
    test_model(model_path)
