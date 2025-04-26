import pywt
import numpy as np
from scipy.io import wavfile

DELIMITER = '1111111111111110'  # 16-bit delimiter which will help in the extraction process

def text_to_bits(text):
    return ''.join(f'{ord(c):08b}' for c in text)

def bits_to_text(bits):
    chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars)

def embed_message(audio_path, output_path, message):
    rate, data = wavfile.read(audio_path)
    if data.ndim > 1:
        data = data[:, 0]  # Use one channel if stereo

    # Normalize data
    data = data / np.max(np.abs(data))

    coeffs = pywt.wavedec(data, 'haar', level=1)
    cA, cD = coeffs

    bits = text_to_bits(message) + DELIMITER
    cA_int = np.round(cA * 1e4).astype(np.int32)

    for i, bit in enumerate(bits):
        if i >= len(cA_int):
            raise ValueError("Message too long to embed.")
        cA_int[i] = (cA_int[i] & ~1) | int(bit)

    cA_embedded = cA_int.astype(np.float32) / 1e4
    coeffs_embedded = [cA_embedded, cD]
    stego_data = pywt.waverec(coeffs_embedded, 'haar')

    # Denormalize
    stego_data = np.int16(stego_data / np.max(np.abs(stego_data)) * 32767)
    wavfile.write(output_path, rate, stego_data)

def extract_message(stego_path):
    rate, data = wavfile.read(stego_path)
    if data.ndim > 1:
        data = data[:, 0]

    data = data / np.max(np.abs(data))
    coeffs = pywt.wavedec(data, 'haar', level=1)
    cA, _ = coeffs
    cA_int = np.round(cA * 1e4).astype(np.int32)

    bits = ''.join(str(x & 1) for x in cA_int)

    end = bits.find(DELIMITER)
    if end != -1:
        bits = bits[:end]
    return bits_to_text(bits)


if __name__ == "__main__":
    audio_path = "D:/Backup/musan/music/fma-western-art/music-fma-wa-0000.wav"
    secret_message = "Xin chao TAKA21"
    output_path = "scripts/traditional_algo/stego_dwt.wav"

    embed_message(audio_path, output_path, secret_message)
    extracted_message = extract_message(output_path)
    print(f"Extracted Message: {extracted_message}")
