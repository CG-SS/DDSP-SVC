#!/usr/bin/env python3
"""
Merge all WAV files in a given directory into a single WAV file if they're smaller than a given threshold.
"""

import argparse
import os
import wave
import contextlib
from pydub import AudioSegment


def get_wav_duration(filepath):
    with contextlib.closing(wave.open(filepath, 'r')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def merge_short_wavs(directory, output_file, threshold_sec=2):
    merged_audio = AudioSegment.silent(duration=0)

    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(".wav"):
            filepath = os.path.join(directory, filename)
            duration = get_wav_duration(filepath)

            if duration < threshold_sec:
                print(f"Merging: {filename} ({duration:.2f}s)")
                audio = AudioSegment.from_wav(filepath)
                merged_audio += audio

    if len(merged_audio) > 0:
        merged_audio.export(output_file, format="wav")
        print(f"\nMerged file saved as: {output_file}")
    else:
        print("No files shorter than threshold were found.")

def main():
    parser = argparse.ArgumentParser(description='Merge WAV files based on a threshold.')
    parser.add_argument('input_dir', type=str, help='Input directory containing .wav files')
    parser.add_argument('output', type=str, help='Output file')
    parser.add_argument('--threshold_seconds', type=float, default=2.0,
                        help='Segment size to be merged. Any audio size smaller than this will be merged. (default: 2.0)')

    args = parser.parse_args()

    merge_short_wavs(args.input_dir, args.output, args.threshold_seconds)

if __name__ == "__main__":
    main()
