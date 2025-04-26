from scipy.io import wavfile
import numpy as np

DELIMITER = '1111111111111110'  # 16-bit delimiter which will help in the extraction process

def message_to_bin(message):
    """Convert a string message to binary."""
    binary_message = ''.join(format(byte, '08b') for byte in message.encode('utf-8'))
    return binary_message

def embed_audio(stego_audio_path, secret_message, cover_audio_path):
    """Embed the secret message into the cover audio."""
    # Load audio file
    rate, samples = wavfile.read(cover_audio_path)

    # Ensure samples are in int16 format
    samples = samples.astype(np.int16)

    # Convert secret message to binary
    binary_message = message_to_bin(secret_message) + DELIMITER

    # Check if the audio can contain the secret message
    if len(binary_message) > len(samples):
        raise ValueError("Message is too long to be hidden in audio")

    # Embed the binary message into the audio's LSB
    for i, bit in enumerate(binary_message):
        samples[i] = (samples[i] & ~1) | int(bit)

    # Save the stego audio
    wavfile.write(stego_audio_path, rate, samples)

def extract_audio(stego_audio_path):
    """Extract the secret message from the stego audio."""
    # Load stego audio file
    rate, samples = wavfile.read(stego_audio_path)

    # Ensure samples are in int16 format
    samples = samples.astype(np.int16)

    # Extract the LSB of each sample as the message bit
    binary_message = ''.join(str(sample & 1) for sample in samples)

    # Extract up to the delimiter
    binary_message = binary_message.split(DELIMITER)[0]

    # Convert the binary message to bytes
    secret_bytes = bytes(int(binary_message[i:i+8], 2) for i in range(0, len(binary_message), 8))

    # Decode the bytes using 'utf-8'
    secret_message = secret_bytes.decode('utf-8')
    return secret_message


if __name__ == "__main__":
    # Example Usage
    cover_audio_path = "D:/Backup/musan/music/fma-western-art/music-fma-wa-0000.wav"
    stego_audio_path = "scripts/traditional_algo/stego_lsb.wav"
    secret_message = "Hello TAKA21"

    # Embed the secret message into the cover audio
    embed_audio(stego_audio_path, secret_message, cover_audio_path)

    print("Extracted message: ", extract_audio(stego_audio_path))
