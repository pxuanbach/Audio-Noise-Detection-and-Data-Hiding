def compare_bits(original_bits, decoded_bits):
    """Compare two bit sequences and print match ratio"""
    # Ensure same length by truncating to shorter sequence
    length = min(len(original_bits), len(decoded_bits))
    original = original_bits[:length]
    decoded = decoded_bits[:length]
    
    # Count matching bits
    matches = sum(1 for i in range(length) if original[i] == decoded[i])
    ratio = matches / length
    
    print(f"Matching bits: {matches}/{length} ({ratio:.2%})")
    return ratio
