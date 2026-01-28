"""
🎵 Music Instrument Detection App - Premium Edition
Advanced Temporal Analysis | ML-Powered | Professional Export
"""

import streamlit as st
import os
import time
import json
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from io import BytesIO

# Authentication
try:
    from auth import check_authentication, show_login_page, show_user_info
    AUTH_ENABLED = True
except ImportError:
    AUTH_ENABLED = False

# Import utilities
from utils import (
    load_and_preprocess_audio,
    compute_mel_spectrogram,
    load_model,
    load_metadata,
    load_thresholds,
    predict_temporal,
    plot_waveform,
    plot_mel_spectrogram,
    plot_confidence_bars,
    plot_temporal_detections,
    plot_confidence_heatmap,
    get_device
)
from utils.audio_processing import load_audio_info
from configs.app_config import UI_CONFIG

# Page configuration
st.set_page_config(
    page_title="Instrument Detector - AI Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PERFECTED ML DESIGN ====================
def inject_premium_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* ============ ROOT VARIABLES ============ */
    :root {
        --primary: #00F5FF;
        --primary-glow: #00F5FF;
        --secondary: #FF00FF;
        --accent: #00FF88;
        --success: #00FF88;
        --warning: #FFB800;
        --error: #FF0055;
        
        --bg-dark: #0A0E27;
        --bg-card: #12172E;
        --bg-surface: #1A1F3A;
        --bg-hover: #232947;
        
        --text-primary: #FFFFFF;
        --text-secondary: #A0AEC0;
        --text-muted: #718096;
        
        --gradient-primary: linear-gradient(135deg, #00F5FF 0%, #00B8D4 50%, #0091EA 100%);
        --gradient-accent: linear-gradient(135deg, #00FF88 0%, #00D97E 100%);
        --gradient-pink: linear-gradient(135deg, #FF00FF 0%, #DD00DD 100%);
        --gradient-dark: linear-gradient(180deg, #0A0E27 0%, #12172E 100%);
        
        --glow-cyan: 0 0 20px rgba(0, 245, 255, 0.5), 0 0 40px rgba(0, 245, 255, 0.3);
        --glow-pink: 0 0 20px rgba(255, 0, 255, 0.5), 0 0 40px rgba(255, 0, 255, 0.3);
        --glow-green: 0 0 20px rgba(0, 255, 136, 0.5), 0 0 40px rgba(0, 255, 136, 0.3);
        
        --shadow-lg: 0 10px 40px rgba(0, 0, 0, 0.5);
        --shadow-xl: 0 20px 60px rgba(0, 0, 0, 0.6);
        
        --border-radius: 12px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* ============ BASE LAYOUT ============ */
    * {
        font-family: 'Rajdhani', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: var(--bg-dark);
        color: var(--text-primary);
        padding-bottom: 60px;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* ============ STREAMLIT HEADER ============ */
    header[data-testid="stHeader"] {
        background: rgba(10, 14, 39, 0.95) !important;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(0, 245, 255, 0.1);
    }
    
    /* Sidebar toggle button */
    button[kind="header"] {
        color: var(--primary) !important;
        border: 2px solid var(--primary) !important;
        border-radius: 8px !important;
        padding: 8px !important;
        background: rgba(0, 245, 255, 0.1) !important;
    }
    
    button[kind="header"]:hover {
        background: rgba(0, 245, 255, 0.2) !important;
        box-shadow: var(--glow-cyan) !important;
    }
    
    /* Style file uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 2px dashed var(--primary) !important;
        border-radius: var(--border-radius) !important;
        padding: 2rem !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary) !important;
        background: rgba(0, 245, 255, 0.05) !important;
        box-shadow: var(--glow-cyan) !important;
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary) !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: var(--bg-surface) !important;
        border: 1px solid var(--primary) !important;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: var(--text-secondary) !important;
    }
    
    /* ============ HERO HEADER ============ */
    .hero-header {
        background: var(--bg-card);
        padding: 56px 48px;
        border-radius: 20px;
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(0, 245, 255, 0.2);
        box-shadow: var(--shadow-lg);
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle at 30% 50%, rgba(0, 245, 255, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 70% 80%, rgba(255, 0, 255, 0.1) 0%, transparent 50%);
        animation: glow-rotate 10s linear infinite;
        pointer-events: none;
    }
    
    @keyframes glow-rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
        text-align: center;
    }
    
    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 64px;
        font-weight: 900;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: 2px;
        text-transform: uppercase;
        filter: drop-shadow(0 0 20px rgba(0, 245, 255, 0.5));
    }
    
    .hero-subtitle {
        font-size: 20px;
        color: var(--text-secondary);
        margin-top: 16px;
        font-weight: 500;
        letter-spacing: 1px;
    }
    
    /* ============ SIDEBAR ============ */
    section[data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 2px solid rgba(0, 245, 255, 0.2) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background: var(--bg-card) !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: rgba(0, 245, 255, 0.2) !important;
    }
    
    section[data-testid="stSidebar"] p {
        color: var(--text-secondary) !important;
    }
     section[data-testid="stSidebar"] p {
    color: var(--text-secondary) !important;
}

/* ============ SIDEBAR BUTTON TEXT FIX ============ */
section[data-testid="stSidebar"] .stButton > button {
    color: var(--bg-dark) !important;
    background: linear-gradient(135deg, #00F5FF 0%, #00B8D4 100%) !important;
    border: none !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #00B8D4 0%, #0091EA 100%) !important;
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.4) !important;
}           
    
    /* ============ EXPANDER (DROPDOWN) ICONS ============ */
    .streamlit-expanderHeader {
        background: var(--bg-surface) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        border: 1px solid rgba(0, 245, 255, 0.2) !important;
        padding: 12px 16px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-hover) !important;
        border-color: var(--primary) !important;
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.2);
    }
    
    /* Make expander arrow visible */
    .streamlit-expanderHeader svg {
        fill: var(--primary) !important;
        stroke: var(--primary) !important;
        filter: drop-shadow(0 0 3px var(--primary));
    }
    
    details summary {
        color: var(--text-primary) !important;
    }
    
    /* ============ CARDS ============ */
    .premium-card {
        background: var(--bg-card) !important;
        border-radius: var(--border-radius);
        padding: 32px;
        margin: 24px 0;
        border: 1px solid rgba(0, 245, 255, 0.2);
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--gradient-primary);
        box-shadow: 0 0 10px var(--primary);
    }
    
    .section-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: var(--primary) !important;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
    }
    
    .section-subtitle {
        font-size: 14px;
        color: var(--text-secondary) !important;
        margin-top: -16px;
        margin-bottom: 20px;
    }
    
    /* ============ FILE SUCCESS ============ */
    .file-success {
        background: rgba(0, 255, 136, 0.1);
        border-left: 4px solid var(--success);
        padding: 20px 24px;
        border-radius: 12px;
        margin: 16px 0;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
    }
    
    .file-success-icon {
        font-size: 24px;
        color: var(--success);
        filter: drop-shadow(0 0 5px var(--success));
    }
    
    .file-success-title {
    font-weight: 600;
    color: var(--primary) !important;
    margin-bottom: 4px;
    text-shadow: 0 0 5px rgba(0, 245, 255, 0.3);
    }
    .file-success-size {
    font-size: 13px;
    color: var(--text-secondary) !important;
    font-weight: 500;
    }
    
    /* ============ AUDIO PREVIEW ============ */
    .audio-stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-top: 24px;
    }
    
    .audio-stat-card {
        background: var(--bg-surface);
        border: 1px solid rgba(0, 245, 255, 0.3);
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .audio-stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-primary);
    }
    
    .audio-stat-card:hover {
        background: var(--bg-hover);
        border-color: var(--primary);
        transform: translateY(-4px);
        box-shadow: var(--glow-cyan);
    }
    
    .audio-stat-icon {
        font-size: 36px;
        margin-bottom: 12px;
        filter: drop-shadow(0 0 10px var(--primary));
    }
    
    .audio-stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 32px;
        font-weight: 700;
        color: var(--primary);
        display: block;
        margin-bottom: 6px;
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
    }
    
    .audio-stat-label {
        font-size: 12px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* ============ BUTTONS ============ */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: var(--bg-dark) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 48px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        box-shadow: var(--glow-cyan) !important;
        transition: var(--transition) !important;
        font-family: 'Orbitron', sans-serif !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.7), 0 0 60px rgba(0, 245, 255, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98) !important;
    }
    
    /* ============ PROGRESS ============ */
    .progress-container {
        background: var(--bg-card);
        border-radius: var(--border-radius);
        padding: 24px;
        margin: 24px 0;
        border: 1px solid rgba(0, 245, 255, 0.2);
    }
    
    .progress-title {
        font-size: 16px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .stProgress > div > div {
        background: var(--gradient-primary) !important;
        height: 8px !important;
        border-radius: 4px !important;
        box-shadow: var(--glow-cyan) !important;
    }
    
    .stProgress > div {
        background: var(--bg-surface) !important;
        border-radius: 4px !important;
    }
    
    /* ============ SUCCESS BANNER ============ */
    .success-banner {
        background: rgba(0, 255, 136, 0.1);
        border-left: 4px solid var(--success);
        padding: 20px 28px;
        border-radius: 12px;
        margin: 24px 0;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: var(--glow-green);
    }
    
    .success-icon {
        font-size: 32px;
        color: var(--success);
        filter: drop-shadow(0 0 5px var(--success));
    }
    
    .success-text {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* ============ DETECTION RESULTS ============ */
    .results-header {
        background: var(--bg-card);
        padding: 40px;
        border-radius: var(--border-radius);
        margin-bottom: 24px;
        text-align: center;
        border: 1px solid rgba(0, 245, 255, 0.3);
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .results-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at center, rgba(0, 245, 255, 0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .results-count {
        font-family: 'Orbitron', sans-serif;
        font-size: 72px;
        font-weight: 900;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
        position: relative;
        z-index: 1;
        filter: drop-shadow(0 0 20px rgba(0, 245, 255, 0.5));
    }
    
    .results-label {
        font-size: 16px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        position: relative;
        z-index: 1;
    }
    
    /* ============ INSTRUMENT PILLS ============ */
    .instrument-pills-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 24px 0;
    }
    
    .instrument-pill {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: var(--bg-surface);
        border: 2px solid var(--primary);
        color: var(--text-primary);
        padding: 12px 24px;
        border-radius: 24px;
        font-size: 14px;
        font-weight: 600;
        transition: var(--transition);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .instrument-pill:hover {
        background: rgba(0, 245, 255, 0.1);
        border-color: var(--primary);
        transform: translateY(-2px);
        box-shadow: var(--glow-cyan);
    }
    
    .pill-name {
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .pill-confidence {
        font-family: 'JetBrains Mono', monospace;
        color: var(--primary);
        font-size: 13px;
        font-weight: 700;
    }
    
    /* ============ DATA TABLE ============ */
    .stDataFrame {
        border: 1px solid rgba(0, 245, 255, 0.2) !important;
        border-radius: var(--border-radius) !important;
        overflow: hidden !important;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: var(--bg-card) !important;
    }
    
    /* ============ TABS ============ */
    .stTabs {
        margin-top: 24px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-surface);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(0, 245, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 12px 28px;
        color: var(--text-secondary);
        font-weight: 600;
        border: 1px solid transparent;
        transition: var(--transition);
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 13px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary);
        background: rgba(0, 245, 255, 0.1);
        border-color: rgba(0, 245, 255, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: var(--bg-dark) !important;
        border: none !important;
        box-shadow: var(--glow-cyan) !important;
        font-weight: 700 !important;
    }
    
    /* ============ EXPORT CARDS ============ */
    .export-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 24px;
        margin-top: 24px;
    }
    
    .export-card {
        background: var(--bg-surface);
        border: 2px solid rgba(0, 245, 255, 0.2);
        border-radius: var(--border-radius);
        padding: 28px;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .export-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .export-card:hover::before {
        transform: scaleX(1);
    }
    
    .export-card:hover {
        border-color: var(--primary);
        transform: translateY(-6px);
        box-shadow: var(--glow-cyan);
    }
    
    .export-icon {
        font-size: 52px;
        margin-bottom: 16px;
        filter: drop-shadow(0 0 10px var(--primary));
    }
    
    .export-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .export-description {
        font-size: 14px;
        color: var(--text-secondary);
        margin-bottom: 24px;
        line-height: 1.6;
    }
    
    .stDownloadButton > button {
        background: var(--gradient-accent) !important;
        color: var(--bg-dark) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: var(--transition) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-family: 'Orbitron', sans-serif !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--glow-green) !important;
    }
    
    /* ============ FORM CONTROLS ============ */
    .stSelectbox label,
    .stSlider label {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        font-size: 14px !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background: var(--bg-surface) !important;
        border-color: rgba(0, 245, 255, 0.3) !important;
    }
    
    /* ============ SLIDER (DETECTION SENSITIVITY) ============ */
    .stSlider {
        padding: 10px 0;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: transparent !important;
    }
    
    /* Slider track (background) */
    .stSlider [data-baseweb="slider"] > div {
        background: var(--bg-surface) !important;
        height: 6px !important;
        border-radius: 3px !important;
    }
    
    /* Slider filled track (active part) */
    .stSlider [data-baseweb="slider"] > div > div {
        background: var(--gradient-primary) !important;
    }
    
    /* Slider thumb (the draggable circle) */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--primary) !important;
        width: 20px !important;
        height: 20px !important;
        border: 3px solid var(--bg-dark) !important;
        box-shadow: var(--glow-cyan) !important;
    }
    
    /* Slider value display */
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        background: var(--primary) !important;
        color: var(--bg-dark) !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        border: 2px solid var(--bg-dark) !important;
        box-shadow: var(--glow-cyan) !important;
    }
    
    /* Slider tick marks */
    .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] > div {
        background: var(--primary) !important;
        opacity: 0.5;
    }
    
    /* Make slider labels visible */
    .stSlider [data-baseweb="slider"] p {
        color: var(--text-secondary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
    }
    
    /* ============ INFO/SUCCESS/WARNING/ERROR ============ */
    .stAlert {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        border-left: 4px solid var(--primary) !important;
        color: var(--text-primary) !important;
    }
    
    /* ============ AUDIO PLAYER ============ */
    audio {
        width: 100%;
        border-radius: 8px;
        background: var(--bg-surface);
        border: 1px solid rgba(0, 245, 255, 0.2);
    }
    
    /* ============ DIVIDER ============ */
    hr {
        margin: 32px 0 !important;
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, var(--primary), transparent) !important;
    }
    
    /* ============ RESPONSIVE ============ */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 40px;
        }
        
        .section-title {
            font-size: 22px;
        }
        
        .audio-stats-grid {
            grid-template-columns: 1fr;
        }
        
        .results-count {
            font-size: 56px;
        }
    }
    
    </style>
    """, unsafe_allow_html=True)


# ==================== DATA CLASSES ====================
class PredictionResult:
    """Enhanced prediction result with complete metadata"""
    def __init__(self, audio_path, aggregation, threshold_mult):
        self.audio_path = audio_path
        self.audio_name = Path(audio_path).name
        self.aggregation = aggregation
        self.threshold_multiplier = threshold_mult
        self.timestamp = datetime.now().isoformat()
        self.audio_metadata = {}
        self.detected_instruments = []
        self.all_predictions = {}
        self.temporal_detections = {}
        self.segments = []
        self.window_probs = []
        self.waveform = None
        self.mel_spectrogram = None
        self.total_duration = 0.0
    
    def to_json(self):
        """Export to JSON format"""
        return json.dumps({
            "metadata": {
                "audio_name": self.audio_name,
                "timestamp": self.timestamp,
                "aggregation": self.aggregation,
                "threshold_multiplier": self.threshold_multiplier,
                "sample_rate": self.audio_metadata.get('sample_rate'),
                "duration": self.audio_metadata.get('duration'),
            },
            "detected_instruments": self.detected_instruments,
            "all_predictions": {
                inst: {
                    "probability": float(pred['probability']),
                    "threshold": float(pred['threshold']),
                    "detected": bool(pred['detected'])
                }
                for inst, pred in self.all_predictions.items()
            },
            "temporal_detections": self.temporal_detections,
        }, indent=2)


# ==================== PDF GENERATION ====================
def generate_pdf_report(prediction_result, plots_data):
    """Generate professional PDF report with visualizations"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph, 
            Spacer, Image, PageBreak
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            topMargin=0.5*inch, 
            bottomMargin=0.5*inch
        )
        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#00F5FF'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#1A1F3A'),
            spaceAfter=12,
            spaceBefore=16,
            fontName='Helvetica-Bold'
        )

        # Title
        story.append(Paragraph("🎵 Instrument Detection Report", title_style))
        story.append(Spacer(1, 0.3*inch))

        # Metadata section
        story.append(Paragraph("Report Information", heading_style))
        metadata_data = [
            ['Audio File:', prediction_result.audio_name],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Detection Mode:', f"{prediction_result.aggregation.upper()} Aggregation"],
            ['Duration:', f"{prediction_result.audio_metadata.get('duration', 0):.2f} seconds"],
            ['Sample Rate:', f"{prediction_result.audio_metadata.get('sample_rate', 0):,} Hz"],
            ['Threshold Multiplier:', f"{prediction_result.threshold_multiplier}x"],
        ]

        metadata_table = Table(metadata_data, colWidths=[2.5*inch, 3.5*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.4*inch))

        # Detection Summary
        story.append(Paragraph("Detection Summary", heading_style))
        if prediction_result.detected_instruments:
            count = len(prediction_result.detected_instruments)
            instruments = ", ".join([d['instrument'].replace('_', ' ').title() 
                                   for d in prediction_result.detected_instruments])
            summary = f"<b>{count}</b> instrument(s) detected: {instruments}"
            story.append(Paragraph(summary, styles['Normal']))
        else:
            story.append(Paragraph("No instruments detected above threshold", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

        # Detected Instruments Table
        if prediction_result.detected_instruments:
            story.append(Paragraph("Detailed Detection Results", heading_style))
            instrument_data = [['Instrument', 'Confidence', 'Status']]
            
            for det in prediction_result.detected_instruments:
                instrument_data.append([
                    det['instrument'].replace('_', ' ').title(),
                    f"{det['confidence']:.1%}",
                    '✓ Detected'
                ])

            instrument_table = Table(instrument_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            instrument_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00F5FF')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTSIZE', (0, 1), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F9FAFB'), colors.white])
            ]))
            story.append(instrument_table)
            story.append(Spacer(1, 0.4*inch))

        # Temporal Detections
        if prediction_result.temporal_detections:
            story.append(PageBreak())
            story.append(Paragraph("Temporal Analysis", heading_style))
            
            for inst, segments in prediction_result.temporal_detections.items():
                inst_name = inst.replace('_', ' ').title()
                story.append(Paragraph(
                    f"<b>{inst_name}</b> — {len(segments)} detection(s)", 
                    styles['Normal']
                ))
                
                segment_data = [['#', 'Start Time', 'End Time', 'Duration', 'Confidence']]
                for i, seg in enumerate(segments, 1):
                    duration = seg['end'] - seg['start']
                    segment_data.append([
                        str(i),
                        f"{seg['start']:.2f}s",
                        f"{seg['end']:.2f}s",
                        f"{duration:.2f}s",
                        f"{seg['confidence']:.1%}"
                    ])

                segment_table = Table(segment_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
                segment_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3F4F6')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(segment_table)
                story.append(Spacer(1, 0.3*inch))

        # Visualizations
        if plots_data:
            story.append(PageBreak())
            story.append(Paragraph("Visualizations", heading_style))
            
            for plot_name, plot_img in plots_data.items():
                if plot_img:
                    img = Image(plot_img, width=6.5*inch, height=3.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.3*inch))

        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer_text = "Report generated by Instrument Detector | Powered by EfficientNet-B0 Deep Learning"
        story.append(Paragraph(
            footer_text,
            ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.grey,
                alignment=TA_CENTER
            )
        ))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

    except ImportError:
        st.error("⚠️ ReportLab not installed. Install with: `pip install reportlab`")
        return None
    except Exception as e:
        st.error(f"❌ PDF generation error: {str(e)}")
        return None


# ==================== SESSION STATE ====================
def init_session_state():
    """Initialize session state variables"""
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'pdf_generated' not in st.session_state:
        st.session_state.pdf_generated = False
    if 'pdf_buffer' not in st.session_state:
        st.session_state.pdf_buffer = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False


@st.cache_resource
def load_model_cached():
    """Load model with caching"""
    device = get_device()
    return load_model(device), load_metadata(), load_thresholds(), device


# ==================== UI COMPONENTS ====================
def display_hero_header():
    """Display hero header"""
    st.markdown("""
    <div class="hero-header">
        <div class="hero-content">
            <h1 class="hero-title">🎵 MUSIC INSTRUMENT DETECTOR</h1>
            <p class="hero-subtitle">AI-Powered Music Instrument Detection & Analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Enhanced sidebar with configuration"""
    with st.sidebar:
        if AUTH_ENABLED:
            show_user_info()
            st.markdown("---")
        
        st.markdown("### ⚙️ Configuration")
        
        aggregation = st.selectbox(
            "Aggregation Strategy",
            ["max", "mean", "vote"],
            help="How to combine predictions across time windows"
        )
        
        threshold_mult = st.slider(
            "Detection Sensitivity",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Lower = more detections, Higher = stricter filtering"
        )
        
        st.markdown("---")
        
        with st.expander("ℹ️ Model Information"):
            st.markdown("""
            **Architecture:** EfficientNet-B0  
            **Mode:** Temporal Analysis  
            **Classes:** 20 Instruments  
            **Input:** Mel Spectrograms  
            **Window:** Sliding window analysis
            """)
        
        with st.expander("📋 Supported Instruments"):
            instruments = [
                "Accordion", "Acoustic Guitar", "Bass Drum", "Cello",
                "Clarinet", "Double Bass", "Flute", "Hi-hat",
                "Piano", "Saxophone", "Snare Drum", "Trumpet",
                "Violin", "Voice", "Electric Guitar", "Organ",
                "Tambourine", "Trombone", "Tuba", "Xylophone"
            ]
            for inst in instruments:
                st.markdown(f"• {inst}")
        
        st.markdown("---")
        
        if st.button("🔄 Reset Analysis", use_container_width=True, key="reset_sidebar"):
            st.session_state.prediction_result = None
            st.session_state.pdf_generated = False
            st.session_state.pdf_buffer = None
            st.session_state.analysis_complete = False
            st.rerun()
        
        return aggregation, threshold_mult


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory"""
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    path = temp_dir / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(path)


def run_prediction(audio_path, model, thresholds, device, aggregation, threshold_mult):
    """Execute prediction pipeline with progress tracking"""
    result = PredictionResult(audio_path, aggregation, threshold_mult)
    result.audio_metadata = load_audio_info(audio_path)
    
    st.markdown("""
    <div class="progress-container">
        <div class="progress-title">
            🔄 Analyzing audio file...
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    progress = st.progress(0)
    
    try:
        # Load audio
        progress.progress(20)
        time.sleep(0.3)
        y = load_and_preprocess_audio(audio_path)
        
        # Compute spectrogram
        progress.progress(40)
        time.sleep(0.3)
        mel = compute_mel_spectrogram(y)
        result.waveform = y
        result.mel_spectrogram = mel
        
        # Run prediction
        progress.progress(70)
        time.sleep(0.3)
        raw = predict_temporal(audio_path, model, thresholds, device, aggregation, threshold_mult)
        
        # Process results
        progress.progress(90)
        result.detected_instruments = raw['detected_instruments']
        result.all_predictions = raw['all_predictions']
        
        if 'temporal_detections' in raw:
            result.temporal_detections = raw['temporal_detections']
        if 'window_times' in raw:
            result.segments = [{"start": float(s), "end": float(e)} for s, e in raw['window_times']]
        if 'window_probs' in raw:
            result.window_probs = raw['window_probs']
        if 'total_duration' in raw:
            result.total_duration = raw['total_duration']
        
        progress.progress(100)
        time.sleep(0.3)
        progress.empty()
        
        st.markdown("""
        <div class="success-banner">
            <span class="success-icon">✓</span>
            <span class="success-text">Detection completed successfully!</span>
        </div>
        """, unsafe_allow_html=True)
        
        return result
        
    except Exception as e:
        progress.empty()
        st.error(f"❌ Error during detection: {str(e)}")
        return None


def create_timeline_dataframe(result):
    """Create timeline DataFrame from temporal detections"""
    if not result.temporal_detections:
        return None
    
    timeline_data = []
    for instrument, detections in result.temporal_detections.items():
        for detection in detections:
            duration = detection['end'] - detection['start']
            timeline_data.append({
                'Instrument': instrument.replace('_', ' ').title(),
                'Start (s)': f"{detection['start']:.2f}",
                'End (s)': f"{detection['end']:.2f}",
                'Duration (s)': f"{duration:.2f}",
                'Confidence': f"{detection['confidence']:.1%}"
            })
    
    if timeline_data:
        df = pd.DataFrame(timeline_data)
        df['_sort_key'] = df['Start (s)'].astype(float)
        df = df.sort_values('_sort_key').drop('_sort_key', axis=1)
        return df
    
    return None


# ==================== MAIN APPLICATION ====================
def main():
    """Main application logic"""
    
    # Authentication
    if AUTH_ENABLED:
        if not check_authentication():
            show_login_page()
            return
    
    # Inject CSS
    inject_premium_css()
    
    # Initialize session
    init_session_state()
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        try:
            model, metadata, thresholds, device = load_model_cached()
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.stop()
    
    # Display header
    display_hero_header()
    
    # Sidebar configuration
    aggregation, threshold_mult = display_sidebar()
    
    # ==================== UPLOAD SECTION ====================
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📁 Upload Audio File</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Supported formats: WAV, MP3, FLAC, OGG, M4A</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose audio file",
        type=UI_CONFIG['supported_formats'],
        label_visibility="visible"
    )
    
    if uploaded_file:
        file_size = uploaded_file.size / 1024
        st.markdown(f"""
        <div class="file-success">
            <span class="file-success-icon">✓</span>
            <div class="file-success-content">
                <div class="file-success-title">{uploaded_file.name}</div>
                <div class="file-success-size">{file_size:.1f} KB</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== AUDIO PREVIEW ====================
    if uploaded_file:
        audio_path = save_uploaded_file(uploaded_file)
        
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎧 Audio Preview</div>', unsafe_allow_html=True)
        
        # Audio player
        st.audio(audio_path, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        # Audio stats
        info = load_audio_info(audio_path)
        st.markdown(f"""
        <div class="audio-stats-grid">
            <div class="audio-stat-card">
                <div class="audio-stat-icon">🎵</div>
                <span class="audio-stat-value">{info['sample_rate']:,}</span>
                <span class="audio-stat-label">Sample Rate (Hz)</span>
            </div>
            <div class="audio-stat-card">
                <div class="audio-stat-icon">📊</div>
                <span class="audio-stat-value">{info['channels']}</span>
                <span class="audio-stat-label">Channels</span>
            </div>
            <div class="audio-stat-card">
                <div class="audio-stat-icon">⏱️</div>
                <span class="audio-stat-value">{info['duration']:.1f}</span>
                <span class="audio-stat-label">Duration (s)</span>
            </div>
            <div class="audio-stat-card">
                <div class="audio-stat-icon">💾</div>
                <span class="audio-stat-value">{file_size:.1f}</span>
                <span class="audio-stat-label">File Size (KB)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ==================== ANALYZE BUTTON ====================
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔍 Detect Instruments", type="primary", use_container_width=True, key="analyze_btn"):
                result = run_prediction(audio_path, model, thresholds, device, aggregation, threshold_mult)
                if result:
                    st.session_state.prediction_result = result
                    st.session_state.analysis_complete = True
                    st.session_state.pdf_generated = False
                    st.session_state.pdf_buffer = None
                    st.rerun()
    
    # ==================== RESULTS SECTION ====================
    if st.session_state.prediction_result and st.session_state.analysis_complete:
        result = st.session_state.prediction_result
        
        # Detection Results Header
        detected_count = len(result.detected_instruments)
        
        st.markdown(f"""
        <div class="results-header">
            <div class="results-count">{detected_count}</div>
            <div class="results-label">Instrument(s) Detected</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detection Results Card
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 Detection Results</div>', unsafe_allow_html=True)
        
        if result.detected_instruments:
            # Instrument pills
            st.markdown('<div class="instrument-pills-container">', unsafe_allow_html=True)
            for det in result.detected_instruments:
                st.markdown(f"""
                <div class="instrument-pill">
                    <span class="pill-name">{det["instrument"]}</span>
                    <span class="pill-confidence">{det["confidence"]:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Results table
            df = pd.DataFrame(result.detected_instruments)
            df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
            df.columns = ['Instrument', 'Confidence']
            
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No instruments detected above the current threshold. Try adjusting the sensitivity in the sidebar.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ==================== TEMPORAL TIMELINE ====================
        timeline_df = create_timeline_dataframe(result)
        if timeline_df is not None:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⏱️ Temporal Timeline</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">When each instrument appears in the audio</div>', unsafe_allow_html=True)
            
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ==================== VISUALIZATIONS ====================
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Visualizations</div>', unsafe_allow_html=True)
        
        tabs = st.tabs(["📈 Confidence Scores", "🌊 Waveform & Spectrogram", "⏰ Temporal Analysis"])
        
        with tabs[0]:
            st.plotly_chart(
                plot_confidence_bars(result.all_predictions, top_n=None),
                use_container_width=True,
                key="confidence_chart"
            )
        
        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_waveform(result.waveform),
                    use_container_width=True,
                    key="waveform_chart"
                )
            with col2:
                st.plotly_chart(
                    plot_mel_spectrogram(result.mel_spectrogram),
                    use_container_width=True,
                    key="spectrogram_chart"
                )
        
        with tabs[2]:
            if result.temporal_detections:
                st.plotly_chart(
                    plot_temporal_detections(
                        result.temporal_detections,
                        result.total_duration,
                        list(thresholds.keys())
                    ),
                    use_container_width=True,
                    key="timeline_chart"
                )
                
                if result.window_probs:
                    times = [(s['start'], s['end']) for s in result.segments]
                    st.plotly_chart(
                        plot_confidence_heatmap(
                            result.window_probs,
                            times,
                            list(thresholds.keys())
                        ),
                        use_container_width=True,
                        key="heatmap_chart"
                    )
            else:
                st.info("ℹ️ No temporal detections available")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ==================== EXPORT SECTION ====================
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💾 Export Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Download your detection results in multiple formats</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # JSON Export
        with col1:
            st.markdown("""
            <div class="export-card">
                <div class="export-icon">📄</div>
                <div class="export-title">JSON Report</div>
                <div class="export-description">
                    Machine-readable format with complete detection data, 
                    perfect for further processing or integration.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            json_data = result.to_json()
            st.download_button(
                "📥 Download JSON",
                data=json_data,
                file_name=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                key="download_json"
            )
        
        # PDF Export
        with col2:
            st.markdown("""
            <div class="export-card">
                <div class="export-icon">📑</div>
                <div class="export-title">PDF Report</div>
                <div class="export-description">
                    Professional report with visualizations and detailed 
                    detection results, ready to share or present.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🎨 Generate PDF Report", use_container_width=True, key="generate_pdf"):
                with st.spinner("Generating professional PDF report..."):
                    try:
                        import plotly.io as pio
                        
                        plots_data = {}
                        # Add confidence chart
                        fig_conf = plot_confidence_bars(result.all_predictions, top_n=10)
                        plots_data['confidence'] = BytesIO(
                            pio.to_image(fig_conf, format='png', width=900, height=450)
                        )
                        
                        # Generate PDF
                        pdf_buffer = generate_pdf_report(result, plots_data)
                        
                        if pdf_buffer:
                            st.session_state.pdf_generated = True
                            st.session_state.pdf_buffer = pdf_buffer
                            st.success("✓ PDF generated successfully!")
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"PDF generation error: {str(e)}")
            
            if st.session_state.get('pdf_generated') and st.session_state.get('pdf_buffer'):
                st.download_button(
                    "⬇️ Download PDF",
                    data=st.session_state.pdf_buffer,
                    file_name=f"instrument_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()