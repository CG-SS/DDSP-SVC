import argparse
import os
import random
import shutil

import librosa
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import soundfile as sf

from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from logger import utils
from logger.utils import traverse_dir
from reflow.vocoder import Vocoder

_original_load = torch.load


def unsafe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


torch.load = unsafe_load


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    return parser.parse_args(args=args, namespace=namespace)


def process_audio_files(src_dir, out_dir, duration, extensions, sr):
    """
    Process audio files by filtering by extension, converting to WAV, and splitting into chunks.

    Args:
        src_dir (str): Input directory containing audio files
        out_dir (str): Output directory for processed chunks
        duration (float): Duration of each chunk in seconds
        extensions (list): List of file extensions to process (e.g., ['.mp3', '.wav', '.flac'])
    """
    # Create output directory if it doesn't exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Normalize extensions to lowercase with dots
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]

    # Get all files in source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()

            # Check if file extension matches any in the extensions list
            if file_ext in extensions:
                try:
                    # Load audio file
                    y, sr = librosa.load(file_path, sr=None)

                    # Calculate samples per chunk
                    samples_per_chunk = int(duration * sr)

                    # Get base filename without extension
                    base_name = os.path.splitext(os.path.basename(file))[0]

                    # Split audio into chunks
                    for i, start in enumerate(range(0, len(y), samples_per_chunk)):
                        end = min(start + samples_per_chunk, len(y))
                        chunk = y[start:end]

                        # Only save chunks that are at least as long as the specified duration
                        if len(chunk) >= samples_per_chunk:
                            # Create output filename
                            chunk_filename = f"{base_name}_chunk_{i:04d}.wav"
                            chunk_path = os.path.join(out_dir, chunk_filename)

                            # Save chunk as WAV file
                            sf.write(chunk_path, chunk, sr)

                except Exception:
                    # Skip files that can't be processed
                    print(f"Error processing {file_path}")
                    continue

def preprocess(src_path, output_path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size, device = 'cuda', use_pitch_aug = False, extensions = ['wav'], duration = 2.0):

    path_srcdir  = os.path.join(src_path, 'audio')

    audio_output_path = os.path.join(output_path, 'audio')
    process_audio_files(path_srcdir, audio_output_path, duration, extensions, sample_rate)

    path_unitsdir  = os.path.join(output_path, 'units')
    path_f0dir  = os.path.join(output_path, 'f0')
    path_volumedir  = os.path.join(output_path, 'volume')
    path_augvoldir  = os.path.join(output_path, 'aug_vol')
    path_meldir  = os.path.join(output_path, 'mel')
    path_augmeldir  = os.path.join(output_path, 'aug_mel')
    path_skipdir = os.path.join(output_path, 'skip')
    
    # list files
    filelist =  traverse_dir(
        audio_output_path,
        extensions=extensions,
        is_pure=True,
        is_sort=True,
        is_ext=True)
    
    # pitch augmentation dictionary
    pitch_aug_dict = {}
    
    # run  
    def process(file):
        binfile = file+'.npy'
        path_srcfile = os.path.join(audio_output_path, file)
        path_unitsfile = os.path.join(path_unitsdir, binfile)
        path_f0file = os.path.join(path_f0dir, binfile)
        path_volumefile = os.path.join(path_volumedir, binfile)
        path_augvolfile = os.path.join(path_augvoldir, binfile)
        path_melfile = os.path.join(path_meldir, binfile)
        path_augmelfile = os.path.join(path_augmeldir, binfile)
        path_skipfile = os.path.join(path_skipdir, file)
        
        # load audio
        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        audio_t = torch.from_numpy(audio).float().to(device)
        audio_t = audio_t.unsqueeze(0)
        
        # extract volume
        volume = volume_extractor.extract(audio)
        
        # extract mel and volume augmentaion
        if mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_t, sample_rate)
            mel = mel_t.squeeze().to('cpu').numpy()
            
            max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
            max_shift = min(1, np.log10(1/max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            if use_pitch_aug:
                keyshift = random.uniform(-5, 5)
            else:
                keyshift = 0
            
            aug_mel_t = mel_extractor.extract(audio_t * (10 ** log10_vol_shift), sample_rate, keyshift = keyshift)
            aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
            aug_vol = volume_extractor.extract(audio * (10 ** log10_vol_shift))
            
        # units encode
        units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
        units = units_t.squeeze().to('cpu').numpy()
        
        # extract f0
        f0 = f0_extractor.extract(audio, uv_interp = False)
        
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

            # save npy     
            os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
            np.save(path_unitsfile, units)
            os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
            np.save(path_f0file, f0)
            os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
            np.save(path_volumefile, volume)
            if mel_extractor is not None:
                pitch_aug_dict[file] = keyshift
                os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
                np.save(path_melfile, mel)
                os.makedirs(os.path.dirname(path_augmelfile), exist_ok=True)
                np.save(path_augmelfile, aug_mel)
                os.makedirs(os.path.dirname(path_augvolfile), exist_ok=True)
                np.save(path_augvolfile, aug_vol)
        else:
            print('\n[Error] F0 extraction failed: ' + path_srcfile)
            os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
            shutil.move(path_srcfile, os.path.dirname(path_skipfile))
            print('This file has been moved to ' + path_skipfile)
    print('Preprocess the audio clips in :', audio_output_path)
    
    # single process
    for file in tqdm(filelist, total=len(filelist)):
        process(file)
    
    if mel_extractor is not None:
        path_pitchaugdict = os.path.join(output_path, 'pitch_aug_dict.npy')
        np.save(path_pitchaugdict, pitch_aug_dict)
    # multi-process (have bugs)
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(process, filelist), total=len(filelist)))
    '''
                
if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load config
    args = utils.load_config(cmd.config)
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    
    extensions = args.data.extensions
    
    # initialize f0 extractor
    f0_extractor = F0_Extractor(
                        args.data.f0_extractor, 
                        args.data.sampling_rate, 
                        args.data.block_size, 
                        args.data.f0_min, 
                        args.data.f0_max)
    
    # initialize volume extractor
    volume_extractor = Volume_Extractor(args.data.block_size, args.data.volume_smooth_size)
    
    # initialize mel extractor
    use_pitch_aug = False
    mel_extractor = Vocoder(args.vocoder.type, args.vocoder.ckpt, device = device)
    if mel_extractor.vocoder_sample_rate != sample_rate or mel_extractor.vocoder_hop_size != hop_size:
        mel_extractor = None
        print('Unmatch vocoder parameters, mel extraction is ignored!')
    elif args.model.use_pitch_aug:
        use_pitch_aug = True
    
    # initialize units encoder
    if args.data.encoder == 'cnhubertsoftfish':
        cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10
    units_encoder = Units_Encoder(
                        args.data.encoder, 
                        args.data.encoder_ckpt, 
                        args.data.encoder_sample_rate, 
                        args.data.encoder_hop_size,
                        cnhubertsoft_gate=cnhubertsoft_gate,
                        device = device)
    
    # preprocess training set
    preprocess(args.data.train_path, os.path.join(args.env.expdir, 'train'), f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size, device = device, use_pitch_aug = use_pitch_aug, extensions = extensions, duration = args.data.duration)
    
    # preprocess validation set
    preprocess(args.data.valid_path, os.path.join(args.env.expdir, 'val'), f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size, device = device, use_pitch_aug = False, extensions = extensions, duration = args.data.duration)
    
