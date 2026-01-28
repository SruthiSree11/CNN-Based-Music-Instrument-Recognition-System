"""
Visualization utilities for plotting.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from configs.app_config import INSTRUMENT_COLORS, AUDIO_CONFIG


def plot_waveform(y, sr=AUDIO_CONFIG['sr']):
    """
    Plot audio waveform using Plotly.
    
    Args:
        y: Audio array
        sr: Sample rate
    
    Returns:
        Plotly figure
    """
    time = np.linspace(0, len(y) / sr, len(y))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=y,
        mode='lines',
        line=dict(color='#4ECDC4', width=1),
        name='Waveform'
    ))
    
    fig.update_layout(
        title='Audio Waveform',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        template='plotly_dark',
        height=300,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def plot_mel_spectrogram(mel, sr=AUDIO_CONFIG['sr'], hop_length=AUDIO_CONFIG['hop_length']):
    """
    Plot mel spectrogram using Plotly.
    
    Args:
        mel: Mel spectrogram
        sr: Sample rate
        hop_length: Hop length
    
    Returns:
        Plotly figure
    """
    # Create time and frequency axes
    times = librosa.frames_to_time(np.arange(mel.shape[1]), sr=sr, hop_length=hop_length)
    freqs = librosa.mel_frequencies(n_mels=mel.shape[0], fmin=0, fmax=sr/2)
    
    fig = go.Figure(data=go.Heatmap(
        z=mel,
        x=times,
        y=freqs,
        colorscale='Viridis',
        colorbar=dict(title='dB')
    ))
    
    fig.update_layout(
        title='Mel Spectrogram',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        template='plotly_dark',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def plot_confidence_bars(predictions, top_n=None):
    """
    Plot confidence bars for all instruments.
    
    Args:
        predictions: Dictionary with predictions
        top_n: Show only top N instruments (None for all)
    
    Returns:
        Plotly figure
    """
    # Extract data
    instruments = []
    confidences = []
    colors = []
    detected = []
    
    for inst, pred in predictions.items():
        instruments.append(inst)
        confidences.append(pred['probability'])
        colors.append(INSTRUMENT_COLORS.get(inst, '#CCCCCC'))
        detected.append(pred['detected'])
    
    # Sort by confidence
    sorted_indices = np.argsort(confidences)[::-1]
    
    if top_n is not None:
        sorted_indices = sorted_indices[:top_n]
    
    instruments = [instruments[i] for i in sorted_indices]
    confidences = [confidences[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    detected = [detected[i] for i in sorted_indices]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    for i, (inst, conf, color, det) in enumerate(zip(instruments, confidences, colors, detected)):
        fig.add_trace(go.Bar(
            x=[conf],
            y=[inst],
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color='white' if det else 'gray', width=2 if det else 1)
            ),
            name=inst,
            showlegend=False,
            hovertemplate=f'<b>{inst}</b><br>Confidence: {conf:.3f}<br>{"✓ Detected" if det else "✗ Not detected"}<extra></extra>'
        ))
    
    # Add threshold lines
    for inst, pred in predictions.items():
        if inst in instruments:
            fig.add_shape(
                type='line',
                x0=pred['threshold'],
                x1=pred['threshold'],
                y0=instruments.index(inst) - 0.4,
                y1=instruments.index(inst) + 0.4,
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.5
            )
    
    fig.update_layout(
        title='Instrument Confidence Scores',
        xaxis_title='Confidence',
        yaxis_title='',
        template='plotly_dark',
        height=max(400, len(instruments) * 25),
        margin=dict(l=150, r=50, t=50, b=50),
        xaxis=dict(range=[0, 1])
    )
    
    return fig


def plot_temporal_detections(temporal_detections, total_duration, instrument_names):
    """
    Plot temporal detections as a timeline.
    
    Args:
        temporal_detections: Dictionary of instrument -> list of detection segments
        total_duration: Total audio duration in seconds
        instrument_names: List of all instrument names
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Only show instruments that were detected
    detected_instruments = list(temporal_detections.keys())
    
    if not detected_instruments:
        # No detections
        fig.add_annotation(
            text="No instruments detected in temporal analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='gray')
        )
        
        fig.update_layout(
            title='Temporal Detection Timeline',
            template='plotly_dark',
            height=200,
            margin=dict(l=150, r=50, t=50, b=50)
        )
        
        return fig
    
    # Sort instruments by first detection time
    detected_instruments.sort(key=lambda x: temporal_detections[x][0]['start'])
    
    # Plot each instrument's detections
    for i, inst in enumerate(detected_instruments):
        detections = temporal_detections[inst]
        color = INSTRUMENT_COLORS.get(inst, '#CCCCCC')
        
        for det in detections:
            # Add rectangle for detection
            fig.add_trace(go.Scatter(
                x=[det['start'], det['end'], det['end'], det['start'], det['start']],
                y=[i-0.4, i-0.4, i+0.4, i+0.4, i-0.4],
                fill='toself',
                fillcolor=color,
                line=dict(color=color, width=2),
                opacity=det['confidence'],
                hovertemplate=f'<b>{inst}</b><br>Time: {det["start"]:.2f}s - {det["end"]:.2f}s<br>Confidence: {det["confidence"]:.3f}<extra></extra>',
                showlegend=False
            ))
    
    fig.update_layout(
        title='Temporal Detection Timeline',
        xaxis_title='Time (s)',
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(detected_instruments))),
            ticktext=detected_instruments
        ),
        template='plotly_dark',
        height=max(300, len(detected_instruments) * 40),
        margin=dict(l=150, r=50, t=50, b=50),
        xaxis=dict(range=[0, total_duration])
    )
    
    return fig


def create_instrument_timeline(temporal_detections, total_duration):
    """
    Create a compact timeline visualization.
    
    Args:
        temporal_detections: Dictionary of instrument -> list of detection segments
        total_duration: Total audio duration in seconds
    
    Returns:
        Plotly figure
    """
    if not temporal_detections:
        fig = go.Figure()
        fig.add_annotation(
            text="No temporal detections",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color='gray')
        )
        fig.update_layout(
            template='plotly_dark',
            height=100,
            margin=dict(l=50, r=50, t=30, b=30)
        )
        return fig
    
    fig = go.Figure()
    
    row_height = 30
    y_pos = 0
    
    for inst, detections in temporal_detections.items():
        color = INSTRUMENT_COLORS.get(inst, '#CCCCCC')
        
        for det in detections:
            fig.add_shape(
                type='rect',
                x0=det['start'],
                x1=det['end'],
                y0=y_pos,
                y1=y_pos + row_height,
                fillcolor=color,
                opacity=det['confidence'] * 0.8,
                line=dict(color=color, width=1)
            )
        
        # Add label
        fig.add_annotation(
            x=-0.5,
            y=y_pos + row_height/2,
            text=inst,
            showarrow=False,
            xanchor='right',
            font=dict(size=10)
        )
        
        y_pos += row_height + 5
    
    fig.update_layout(
        title='Detection Timeline',
        xaxis_title='Time (s)',
        template='plotly_dark',
        height=len(temporal_detections) * (row_height + 5) + 100,
        margin=dict(l=150, r=50, t=50, b=50),
        xaxis=dict(range=[0, total_duration]),
        yaxis=dict(visible=False),
        showlegend=False
    )
    
    return fig


def plot_confidence_heatmap(window_probs, window_times, instrument_names):
    """
    Plot confidence heatmap over time.
    
    Args:
        window_probs: Array of shape (n_windows, n_classes)
        window_times: List of (start, end) tuples
        instrument_names: List of instrument names
    
    Returns:
        Plotly figure
    """
    window_probs = np.array(window_probs)
    
    # Create time labels
    time_labels = [f"{start:.1f}-{end:.1f}s" for start, end in window_times]
    
    fig = go.Figure(data=go.Heatmap(
        z=window_probs.T,  # Transpose to have instruments on y-axis
        x=time_labels,
        y=instrument_names,
        colorscale='YlOrRd',
        colorbar=dict(title='Confidence'),
        hovertemplate='Instrument: %{y}<br>Time: %{x}<br>Confidence: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Confidence Heatmap Over Time',
        xaxis_title='Time Window',
        yaxis_title='Instrument',
        template='plotly_dark',
        height=max(400, len(instrument_names) * 20),
        margin=dict(l=150, r=50, t=50, b=100),
        xaxis=dict(tickangle=-45)
    )
    
    return fig
