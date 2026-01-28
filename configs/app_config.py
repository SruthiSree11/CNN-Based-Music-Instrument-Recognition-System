"""
Application configuration settings.
"""

# Audio processing settings
AUDIO_CONFIG = {
    'sr': 22050,                    # Sample rate
    'duration': 10.0,               # Duration in seconds
    'n_mels': 128,                  # Number of mel bins
    'n_fft': 2048,                  # FFT window size
    'hop_length': 512,              # Hop length for STFT
    'window_duration': 2.0,         # Sliding window duration (seconds)
    'hop_duration': 1.0,            # Hop between windows (seconds)
}

# Model settings
MODEL_CONFIG = {
    'model_path': 'models/model.pt',
    'metadata_path': 'models/metadata.json',
    'thresholds_path': 'models/thresholds.json',
}

# Inference settings
INFERENCE_CONFIG = {
    'default_aggregation': 'max',   # max, mean, vote
    'default_threshold_mult': 1.0,  # Multiply base thresholds
    'temporal_mode': True,          # Enable temporal detection
}

# UI settings
UI_CONFIG = {
    'max_file_size_mb': 50,
    'supported_formats': ['ogg', 'wav', 'mp3', 'flac', 'm4a'],
    'theme': 'dark',
}

# Instrument colors for visualization
INSTRUMENT_COLORS = {
    'accordion': '#FF6B6B',
    'banjo': '#4ECDC4',
    'bass': '#45B7D1',
    'cello': '#FFA07A',
    'clarinet': '#98D8C8',
    'cymbals': '#F7DC6F',
    'drums': '#BB8FCE',
    'flute': '#85C1E2',
    'guitar': '#52B788',
    'mallet_percussion': '#FFD93D',
    'mandolin': '#6BCB77',
    'organ': '#A8E6CF',
    'piano': '#FF8B94',
    'saxophone': '#FFAAA5',
    'synthesizer': '#B4A7D6',
    'trombone': '#8E7CC3',
    'trumpet': '#FFC857',
    'ukulele': '#6C5B7B',
    'violin': '#C06C84',
    'voice': '#F67280',
}
