import os
import random
import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Global configurations
MAX_SEGMENTS_PER_FILE = 5
MIN_DURATION = 0.3  # seconds
MAX_DURATION = 30.0  # seconds
OUTPUT_DIR = Path("datasets/processed")
MUSAN_DIR = Path("D:/Backup/musan")  # Change this to your MUSAN dataset path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_random_segments(audio, sr, min_duration=MIN_DURATION, max_duration=MAX_DURATION, max_segments=MAX_SEGMENTS_PER_FILE):
    """Create random segments from an audio file"""
    audio_duration = len(audio) / sr
    segments = []

    if audio_duration < min_duration:
        return segments

    # Adjust max_duration if audio is shorter
    max_duration = min(max_duration, audio_duration)

    # Calculate number of segments for this file
    num_segments = random.randint(1, max_segments)

    for _ in range(num_segments):
        # Random duration between min and max
        segment_duration = random.uniform(min_duration, max_duration)
        segment_samples = int(segment_duration * sr)

        # Random start point
        max_start = len(audio) - segment_samples
        if max_start <= 0:
            continue

        start_idx = random.randint(0, max_start)
        segment = audio[start_idx:start_idx + segment_samples]

        segments.append((segment, sr))

    return segments

def process_directory(input_dir, output_dir, category):
    """Process all audio files in a directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) / category
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_dir.rglob("*.wav"))
    logger.info(f"Found {len(audio_files)} files in {category} directory")

    for audio_path in tqdm(audio_files, desc=f"Processing {category}"):
        try:
            # Read audio file
            audio, sr = sf.read(str(audio_path))

            # Create segments
            segments = create_random_segments(audio, sr)

            # Save segments
            for i, (segment, sr) in enumerate(segments):
                output_name = f"{audio_path.stem}_seg{i}.wav"
                output_path = output_dir / output_name
                sf.write(str(output_path), segment, sr)

        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")

def main():
    # Create output directories
    logger.info("Starting audio preprocessing...")

    # Process speech and music directories
    process_directory(MUSAN_DIR / "speech", OUTPUT_DIR, "speech")
    process_directory(MUSAN_DIR / "music", OUTPUT_DIR, "music")

    logger.info("Preprocessing completed!")

if __name__ == "__main__":
    main()
