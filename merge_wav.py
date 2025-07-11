import os
import wave
import contextlib
from pydub import AudioSegment

# Directory containing the .wav files
INPUT_DIR = "./wav_files"  # change this as needed
OUTPUT_FILE = "merged_short.wav"
THRESHOLD_SECONDS = 2


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


if __name__ == "__main__":
    merge_short_wavs(INPUT_DIR, OUTPUT_FILE, THRESHOLD_SECONDS)
