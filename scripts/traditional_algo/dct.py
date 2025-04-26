import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct, idct
import math

def normalize_audio(audio_data):
    """Normalize audio data to range [-1, 1]"""
    return audio_data / np.max(np.abs(audio_data))

def denormalize_audio(normalized_data, original_max):
    """Restore audio data to original value range"""
    return (normalized_data * original_max).astype(np.int16)

def embed_message(audio_path, message, output_path, block_size=1024, alpha=0.1):
    """
    Embed message into audio file using DCT
    :param audio_path: Path to original audio file
    :param message: Message string to hide
    :param output_path: Path to output audio file
    :param block_size: DCT block size
    :param alpha: Embedding strength coefficient
    """
    # Read audio file
    sample_rate, audio_data = wavfile.read(audio_path)

    # Convert message to bit string
    message_bits = ''.join([format(ord(c), '08b') for c in message])
    bits_length = len(message_bits)

    # Normalize audio data
    original_max = np.max(np.abs(audio_data))
    normalized_audio = normalize_audio(audio_data)

    # Divide into blocks
    num_blocks = len(normalized_audio) // block_size
    if bits_length > num_blocks:
        raise ValueError("Message too long for audio file")

    # Process each block
    for i in range(num_blocks):
        block = normalized_audio[i*block_size:(i+1)*block_size]

        # Perform DCT
        dct_block = dct(block, type=2, norm='ortho')

        # Embed bit if there's still message to embed
        if i < bits_length:
            # Use second DCT coefficient for embedding
            if message_bits[i] == '1':
                dct_block[1] = abs(dct_block[1]) + alpha
            else:
                dct_block[1] = -(abs(dct_block[1]) + alpha)

        # Perform IDCT
        normalized_audio[i*block_size:(i+1)*block_size] = idct(dct_block, type=2, norm='ortho')

    # Restore to original value range
    stego_audio = denormalize_audio(normalized_audio, original_max)

    # Save audio file
    wavfile.write(output_path, sample_rate, stego_audio)

def extract_message(stego_path, message_length, block_size=1024):
    """
    Extract message from audio file
    :param stego_path: Path to audio file containing hidden message
    :param message_length: Length of message (in characters)
    :param block_size: DCT block size
    :return: Hidden message
    """
    # Read audio file
    sample_rate, stego_audio = wavfile.read(stego_path)

    # Normalize data
    normalized_audio = normalize_audio(stego_audio)

    # Extract bits
    extracted_bits = ""
    bits_length = message_length * 8

    for i in range(bits_length):
        block = normalized_audio[i*block_size:(i+1)*block_size]
        dct_block = dct(block, type=2, norm='ortho')
        extracted_bits += '1' if dct_block[1] > 0 else '0'

    # Convert bit string to characters
    message = ""
    for i in range(0, len(extracted_bits), 8):
        byte = extracted_bits[i:i+8]
        message += chr(int(byte, 2))

    return message

# Example usage
if __name__ == "__main__":
    audio_path = "D:/Backup/musan/music/fma-western-art/music-fma-wa-0000.wav"
    secret_message = "Xin chao TAKA21"
    output_path = "scripts/traditional_algo/stego_dct.wav"

    # Embed message
    embed_message(audio_path, secret_message, output_path)

    # Extract message
    extracted_message = extract_message(output_path, len(secret_message))
    print(f"Extracted message: {extracted_message}")
