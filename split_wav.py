#!/usr/bin/env python3
"""
Split WAV files using Silero VAD (Voice Activity Detection)
Processes all .wav files in input directory and outputs segments to output directory
"""

import argparse
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


def load_silero_vad():
    """Load Silero VAD model"""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    return model, utils


def get_speech_timestamps(audio, model, utils, sampling_rate=16000, threshold=0.5):
    """Get speech timestamps from audio using Silero VAD"""
    (get_speech_timestamps_func,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    speech_timestamps = get_speech_timestamps_func(
        audio,
        model,
        sampling_rate=sampling_rate,
        threshold=threshold,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        window_size_samples=512,
        speech_pad_ms=30
    )

    return speech_timestamps


def merge_close_segments(timestamps, min_silence_samples):
    """Merge segments that are close together based on minimum silence duration"""
    if not timestamps:
        return []

    merged = []
    current = timestamps[0].copy()

    for next_seg in timestamps[1:]:
        silence_duration = next_seg['start'] - current['end']

        if silence_duration < min_silence_samples:
            # Merge segments
            current['end'] = next_seg['end']
        else:
            # Keep segments separate
            merged.append(current)
            current = next_seg.copy()

    merged.append(current)
    return merged


def split_long_segments(segments, audio, max_length_samples, min_length_samples):
    """Split segments that are too long"""
    final_segments = []

    for seg in segments:
        seg_length = seg['end'] - seg['start']

        if seg_length <= max_length_samples:
            final_segments.append(seg)
        else:
            # Split the segment into smaller chunks
            start = seg['start']
            while start < seg['end']:
                end = min(start + max_length_samples, seg['end'])
                if end - start >= min_length_samples:
                    final_segments.append({'start': start, 'end': end})
                start = end

    return final_segments


def process_wav_file(wav_path, output_dir, model, utils, args):
    """Process a single WAV file and save segments"""
    print(f"\nProcessing: {wav_path}")

    # Load audio
    try:
        waveform, sample_rate = torchaudio.load(wav_path)
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        return 0

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz if needed (Silero VAD expects 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform_16k = resampler(waveform)
    else:
        waveform_16k = waveform

    # Get speech timestamps
    audio_16k = waveform_16k.squeeze()
    timestamps = get_speech_timestamps(audio_16k, model, utils, sampling_rate=16000)

    if not timestamps:
        print(f"No speech detected in {wav_path}")
        return 0

    # Convert time-based parameters to samples
    min_silence_samples = int(args.min_silence_duration * 16000)
    min_length_samples = int(args.min_segment_length * 16000)
    max_length_samples = int(args.max_segment_length * 16000)

    # Merge close segments
    merged_segments = merge_close_segments(timestamps, min_silence_samples)

    # Split long segments
    final_segments = split_long_segments(merged_segments, audio_16k, max_length_samples, min_length_samples)

    # Filter out segments that are too short
    final_segments = [seg for seg in final_segments if seg['end'] - seg['start'] >= min_length_samples]

    # Save segments
    base_name = Path(wav_path).stem
    segments_saved = 0

    for i, seg in enumerate(final_segments):
        # Extract segment from original audio (not resampled)
        start_original = int(seg['start'] * sample_rate / 16000)
        end_original = int(seg['end'] * sample_rate / 16000)
        segment_audio = waveform[:, start_original:end_original]

        # Save segment
        output_filename = f"{base_name}_segment_{i + 1:03d}.wav"
        output_path = os.path.join(output_dir, output_filename)

        torchaudio.save(output_path, segment_audio, sample_rate)
        segments_saved += 1

        # Calculate segment duration
        duration = (seg['end'] - seg['start']) / 16000
        print(f"  Saved: {output_filename} (duration: {duration:.2f}s)")

    return segments_saved


def main():
    parser = argparse.ArgumentParser(description='Split WAV files using Silero VAD')
    parser.add_argument('input_dir', type=str, help='Input directory containing .wav files')
    parser.add_argument('output_dir', type=str, help='Output directory for segmented files')
    parser.add_argument('--min_segment_length', type=float, default=3.0,
                        help='Minimum segment length in seconds (default: 3.0)')
    parser.add_argument('--max_segment_length', type=float, default=8.0,
                        help='Maximum segment length in seconds (default: 8.0)')
    parser.add_argument('--min_silence_duration', type=float, default=0.4,
                        help='Minimum silence duration to split segments in seconds (default: 0.4)')

    args = parser.parse_args()

    # Validate directories
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all .wav files
    wav_files = list(Path(args.input_dir).glob('*.wav'))
    if not wav_files:
        print(f"No .wav files found in '{args.input_dir}'")
        return

    print(f"Found {len(wav_files)} .wav files")
    print(f"Parameters:")
    print(f"  Min segment length: {args.min_segment_length}s")
    print(f"  Max segment length: {args.max_segment_length}s")
    print(f"  Min silence duration: {args.min_silence_duration}s")

    # Load Silero VAD model
    print("\nLoading Silero VAD model...")
    model, utils = load_silero_vad()
    model.eval()

    # Process each file
    total_segments = 0
    for wav_file in tqdm(wav_files, desc="Processing files"):
        segments = process_wav_file(str(wav_file), args.output_dir, model, utils, args)
        total_segments += segments

    print(f"\nProcessing complete!")
    print(f"Total segments created: {total_segments}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()