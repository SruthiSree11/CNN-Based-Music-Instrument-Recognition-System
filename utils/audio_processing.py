"""
Audio preprocessing utilities.
"""

import numpy as np
import librosa
from configs.app_config import AUDIO_CONFIG


def load_and_preprocess_audio(audio_path, sr=AUDIO_CONFIG['sr'], duration=AUDIO_CONFIG['duration']):
    """
    Load and preprocess audio file to fixed length.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        duration: Duration in seconds
    
    Returns:
        Preprocessed audio array
    """
    # Load audio
    y, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Normalize
    y = y / (np.abs(y).max() + 1e-8)
    
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=30)
    
    # Fix length
    target_length = int(sr * duration)
    
    if len(y) < target_length:
        # Pad if too short
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        # Trim if too long
        y = y[:target_length]
    
    return y


def compute_mel_spectrogram(y, sr=AUDIO_CONFIG['sr']):
    """
    Compute mel spectrogram from audio.
    
    Args:
        y: Audio array
        sr: Sample rate
    
    Returns:
        Mel spectrogram in dB scale
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=AUDIO_CONFIG['n_mels'],
        n_fft=AUDIO_CONFIG['n_fft'],
        hop_length=AUDIO_CONFIG['hop_length'],
        power=2.0
    )
    
    # Convert to dB
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    return mel_db


def normalize_mel(mel):
    """
    Normalize mel spectrogram.
    
    Args:
        mel: Mel spectrogram
    
    Returns:
        Normalized mel spectrogram
    """
    mean = mel.mean()
    std = mel.std()
    mel_norm = (mel - mean) / (std + 1e-8)
    return mel_norm


def create_sliding_windows(audio_path, 
                          window_duration=AUDIO_CONFIG['window_duration'],
                          hop_duration=AUDIO_CONFIG['hop_duration'],
                          sr=AUDIO_CONFIG['sr']):
    """
    Create sliding windows from audio for temporal detection.
    
    Args:
        audio_path: Path to audio file
        window_duration: Window duration in seconds
        hop_duration: Hop duration in seconds
        sr: Sample rate
    
    Returns:
        List of (mel_spectrogram, start_time, end_time) tuples
    """
    # Load full audio (no duration limit)
    y, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Normalize
    y = y / (np.abs(y).max() + 1e-8)
    
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=30)
    
    window_samples = int(window_duration * sr)
    hop_samples = int(hop_duration * sr)
    
    windows = []
    
    for start in range(0, len(y) - window_samples + 1, hop_samples):
        end = start + window_samples
        clip = y[start:end]
        
        # Compute mel
        mel = compute_mel_spectrogram(clip, sr)
        mel = normalize_mel(mel)
        
        start_time = start / sr
        end_time = end / sr
        
        windows.append((mel, start_time, end_time))
    
    return windows, len(y) / sr  # Return windows and total duration


def load_audio_info(audio_path):
    """
    Get audio file information.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dictionary with audio information
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    duration = len(y) / sr
    
    return {
        'duration': duration,
        'sample_rate': sr,
        'channels': 1 if y.ndim == 1 else y.shape[0],
        'samples': len(y)
    }
