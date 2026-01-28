"""
Inference utilities for prediction.
"""

import numpy as np
import torch
from .audio_processing import (
    load_and_preprocess_audio,
    compute_mel_spectrogram,
    normalize_mel,
    create_sliding_windows
)


def predict_single(audio_path, model, thresholds, device, threshold_multiplier=1.0):
    """
    Predict instruments for entire audio file.
    
    Args:
        audio_path: Path to audio file
        model: Trained model
        thresholds: Dictionary of per-class thresholds
        device: torch device
        threshold_multiplier: Multiply thresholds by this factor
    
    Returns:
        Dictionary with predictions
    """
    # Preprocess audio
    y = load_and_preprocess_audio(audio_path)
    mel = compute_mel_spectrogram(y)
    mel = normalize_mel(mel)
    
    # Convert to tensor
    mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mel_tensor = mel_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(mel_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Get instrument names
    instrument_names = list(thresholds.keys())
    
    # Apply thresholds
    predictions = {}
    detected_instruments = []
    
    for inst, prob in zip(instrument_names, probs):
        threshold = thresholds[inst] * threshold_multiplier
        is_present = prob >= threshold
        
        predictions[inst] = {
            'probability': float(prob),
            'threshold': float(threshold),
            'detected': bool(is_present)
        }
        
        if is_present:
            detected_instruments.append({
                'instrument': inst,
                'confidence': float(prob)
            })
    
    # Sort by confidence
    detected_instruments.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'detected_instruments': detected_instruments,
        'all_predictions': predictions
    }


def predict_temporal(audio_path, model, thresholds, device, 
                    aggregation='max', threshold_multiplier=1.0):
    """
    Predict instruments using sliding window approach.
    
    Args:
        audio_path: Path to audio file
        model: Trained model
        thresholds: Dictionary of per-class thresholds
        device: torch device
        aggregation: How to aggregate predictions ('max', 'mean', 'vote')
        threshold_multiplier: Multiply thresholds by this factor
    
    Returns:
        Dictionary with temporal predictions
    """
    # Create sliding windows
    windows, total_duration = create_sliding_windows(audio_path)
    
    if len(windows) == 0:
        return predict_single(audio_path, model, thresholds, device, threshold_multiplier)
    
    # Get predictions for each window
    all_window_probs = []
    window_times = []
    
    model.eval()
    with torch.no_grad():
        for mel, start_time, end_time in windows:
            # Convert to tensor
            mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mel_tensor = mel_tensor.to(device)
            
            # Predict
            logits = model(mel_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            all_window_probs.append(probs)
            window_times.append((start_time, end_time))
    
    all_window_probs = np.array(all_window_probs)  # Shape: (n_windows, n_classes)
    
    # Aggregate predictions
    aggregated_probs = aggregate_predictions(all_window_probs, aggregation)
    
    # Get instrument names
    instrument_names = list(thresholds.keys())
    
    # Apply thresholds to aggregated predictions
    predictions = {}
    detected_instruments = []
    
    for inst, prob in zip(instrument_names, aggregated_probs):
        threshold = thresholds[inst] * threshold_multiplier
        is_present = prob >= threshold
        
        predictions[inst] = {
            'probability': float(prob),
            'threshold': float(threshold),
            'detected': bool(is_present)
        }
        
        if is_present:
            detected_instruments.append({
                'instrument': inst,
                'confidence': float(prob)
            })
    
    # Sort by confidence
    detected_instruments.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Create temporal detections for each instrument
    temporal_detections = {}
    
    for i, inst in enumerate(instrument_names):
        threshold = thresholds[inst] * threshold_multiplier
        
        detections = []
        for j, (start, end) in enumerate(window_times):
            if all_window_probs[j, i] >= threshold:
                detections.append({
                    'start': float(start),
                    'end': float(end),
                    'confidence': float(all_window_probs[j, i])
                })
        
        if detections:
            temporal_detections[inst] = detections
    
    return {
        'detected_instruments': detected_instruments,
        'all_predictions': predictions,
        'temporal_detections': temporal_detections,
        'window_times': window_times,
        'window_probs': all_window_probs.tolist(),
        'total_duration': total_duration
    }


def aggregate_predictions(probs, method='max'):
    """
    Aggregate predictions from multiple windows.
    
    Args:
        probs: Array of shape (n_windows, n_classes)
        method: Aggregation method ('max', 'mean', 'vote')
    
    Returns:
        Aggregated probabilities of shape (n_classes,)
    """
    if method == 'max':
        return np.max(probs, axis=0)
    elif method == 'mean':
        return np.mean(probs, axis=0)
    elif method == 'vote':
        # Majority vote (using 0.5 threshold)
        votes = (probs >= 0.5).astype(int)
        return np.mean(votes, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
