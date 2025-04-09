import zlib
from reedsolo import RSCodec
import numpy as np

# Initialize Reed-Solomon codec with 250 error correction bytes
rs = RSCodec(250)

def text_to_bits(text):
    """Convert text to a list of ints in {0, 1}"""
    return bytearray_to_bits(text_to_bytearray(text))


def bits_to_text(bits: str):
    """Convert a binary string to text"""
    # Chuyển chuỗi nhị phân thành một danh sách các số nguyên (0 hoặc 1)
    bit_list = [int(b) for b in bits]

    # Sử dụng hàm bits_to_bytearray để chuyển danh sách bit thành bytearray
    return bytearray_to_text(bits_to_bytearray(bit_list))


def bytearray_to_bits(x):
    """Convert bytearray to a list of bits"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])

    return result


def bits_to_bytearray(bits):
    """Convert a list of bits to a bytearray"""
    ints = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))

    return bytearray(ints)


def text_to_bytearray(text):
    """Compress and add error correction"""
    assert isinstance(text, str), "expected a string"
    x = zlib.compress(text.encode("utf-8"))
    x = rs.encode(bytearray(x))

    # print("text_to_bytearray", x)
    return x


def bytearray_to_text(x):
    """Apply error correction and decompress"""
    # print(x)
    try:
        text = rs.decode(x)
        # print("bytearray_to_text", text)
        if isinstance(text, tuple):
            text = text[0]
        text = zlib.decompress(text)
        return text.decode("utf-8")
    except BaseException as e:
        # print("Error decoding message", str(e))
        return False
