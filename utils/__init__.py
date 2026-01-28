"""
Utilities for audio processing, model loading, and visualization.
"""

from .audio_processing import (
    load_and_preprocess_audio,
    compute_mel_spectrogram,
    normalize_mel,
    create_sliding_windows
)

from .model_loader import (
    load_model,
    load_metadata,
    load_thresholds,
    get_device
)

from .inference import (
    predict_single,
    predict_temporal,
    aggregate_predictions
)

from .visualization import (
    plot_waveform,
    plot_mel_spectrogram,
    plot_confidence_bars,
    plot_temporal_detections,
    create_instrument_timeline,
    plot_confidence_heatmap
)

__all__ = [
    'load_and_preprocess_audio',
    'compute_mel_spectrogram',
    'normalize_mel',
    'create_sliding_windows',
    'load_model',
    'load_metadata',
    'load_thresholds',
    'get_device',
    'predict_single',
    'predict_temporal',
    'aggregate_predictions',
    'plot_waveform',
    'plot_mel_spectrogram',
    'plot_confidence_bars',
    'plot_temporal_detections',
    'create_instrument_timeline',
    'plot_confidence_heatmap',
]