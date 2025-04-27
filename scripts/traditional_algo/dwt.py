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
    original_data = data.copy()
    if data.ndim > 1:
        data = data[:, 0]

    # Scale to [-1,1] range instead of full normalization
    data = data.astype(np.float32) / 32768.0

    coeffs = pywt.wavedec(data, 'haar', level=1)
    cA, cD = coeffs

    bits = text_to_bits(message) + DELIMITER
    print("Message bits length:", len(text_to_bits(message)))

    # Use smaller scaling factor and control embedding strength
    scale = 1e3
    cD_int = np.round(cD * scale).astype(np.int32)

    for i, bit in enumerate(bits):
        if i >= len(cD_int):
            raise ValueError("Message too long to embed.")
        # Apply controlled modification
        if int(bit) != (cD_int[i] & 1):
            cD_int[i] += (-1 if cD_int[i] & 1 else 1)

    cD_embedded = cD_int.astype(np.float32) / scale
    coeffs_embedded = [cA, cD_embedded]
    stego_data = pywt.waverec(coeffs_embedded, 'haar')

    # Scale back to 16-bit range with controlled amplitude
    stego_data = np.clip(stego_data * 32768.0, -32768, 32767).astype(np.int16)

    # Calculate and print SNR
    snr = 10 * np.log10(np.sum(original_data**2) / np.sum((original_data - stego_data)**2))
    print(f"SNR: {snr:.2f} dB")

    wavfile.write(output_path, rate, stego_data)
    return snr

def extract_message(stego_path):
    rate, data = wavfile.read(stego_path)
    if data.ndim > 1:
        data = data[:, 0]

    # Scale to [-1,1] range
    data = data.astype(np.float32) / 32768.0

    coeffs = pywt.wavedec(data, 'haar', level=1)
    _, cD = coeffs

    # Use same scaling as embedding
    scale = 1e3
    cD_int = np.round(cD * scale).astype(np.int32)

    bits = ''.join(str(x & 1) for x in cD_int)
    end = bits.find(DELIMITER)
    if end != -1:
        bits = bits[:end]
    return bits_to_text(bits)


if __name__ == "__main__":
    # audio_path = "D:/Backup/musan/music/fma-western-art/music-fma-wa-0000.wav"
    # audio_path = "D:/Backup/FSDKaggle2018/test/0bb807c0.wav"
    # audio_path = "D:/Backup/FSDKaggle2018/test/6e32975d.wav"
    # audio_path = "D:/Backup/FSDKaggle2018/test/0db65bf4.wav"
    audio_path = "D:/Backup/FSDKaggle2018/test/008afd93.wav"
    secret_message = "Xin chao TAKA27 abah"
    output_path = "scripts/traditional_algo/stego_dwt.wav"

    embed_message(audio_path, output_path, secret_message)
    extracted_message = extract_message(output_path)
    print(f"Extracted Message: {extracted_message}")
