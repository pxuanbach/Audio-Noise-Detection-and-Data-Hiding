import numpy as np
import torch
import torchaudio

import argparse

def load_audio(filepath):
  waveform, sample_rate = torchaudio.load(filepath)
  return waveform.squeeze(0)  # remove channel dim if mono

def compute_srn(cover, generated):
  residual = cover - generated
  signal_power = torch.sum(cover ** 2)
  noise_power = torch.sum(residual ** 2)
  srn = 10 * torch.log10(signal_power / (noise_power + 1e-10))
  return srn.item()

def compute_ncc(cover, generated):
  cover_flat = cover.view(-1)
  generated_flat = generated.view(-1)

  cover_mean = torch.mean(cover_flat)
  generated_mean = torch.mean(generated_flat)

  numerator = torch.sum((cover_flat - cover_mean) * (generated_flat - generated_mean))
  denominator = torch.sqrt(torch.sum((cover_flat - cover_mean)**2) * torch.sum((generated_flat - generated_mean)**2)) + 1e-10
  ncc = numerator / denominator
  return ncc.item()

def main(cover_path, generated_path):
  # Load audio files
  cover_audio = load_audio(cover_path)
  generated_audio = load_audio(generated_path)

  # Ensure same length
  min_len = min(cover_audio.size(0), generated_audio.size(0))
  cover_audio = cover_audio[:min_len]
  generated_audio = generated_audio[:min_len]

  # Compute SRN and NCC
  srn_value = compute_srn(cover_audio, generated_audio)
  ncc_value = compute_ncc(cover_audio, generated_audio)

  print(f"SRN: {srn_value:.2f} dB")
  print(f"NCC: {ncc_value:.4f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute SRN and NCC between two audio files.")
  parser.add_argument('--cover', type=str, required=True, help='Path to the cover/original audio file')
  parser.add_argument('--generated', type=str, required=True, help='Path to the generated/stego audio file')

  args = parser.parse_args()

  main(args.cover, args.generated)
