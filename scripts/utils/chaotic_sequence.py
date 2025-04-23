def generate_chaotic_sequence(key: float, num: int, r: float = 3.81):
    """Generate chaotic sequence using logistic map"""
    x = [0.0] * (num + 1)
    x[0] = key

    result = []
    for i in range(1, num + 1):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
        result.append(x[i])

    return result

def get_segment_positions(audio_len: int, segment_len: int, max_segments: int = 5, key: float = 0.1):
    """Get non-overlapping segment positions using chaotic sequence"""
    # Calculate maximum possible segments
    max_possible = audio_len // segment_len
    num_segments = min(max_segments, max_possible)

    # Generate chaotic sequence
    sequence = generate_chaotic_sequence(key, num_segments)

    # Convert to positions ensuring no overlap
    available_positions = list(range(0, audio_len - segment_len + 1, segment_len))
    positions = []

    # Use chaotic values to select positions
    for x in sequence:
        if not available_positions:
            break

        # Map chaotic value [0,1] to available position index
        idx = int(x * len(available_positions))
        idx = min(idx, len(available_positions) - 1)  # Ensure valid index
        positions.append(available_positions.pop(idx))

    return sorted(positions)
