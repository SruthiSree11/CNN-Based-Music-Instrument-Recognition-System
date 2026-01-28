"""
Model loading utilities with Hugging Face model hosting.
Downloads model from HF if not present locally.
"""

import json
import torch
import torch.nn as nn
import timm
import os
from pathlib import Path
import streamlit as st
import requests
from configs.app_config import MODEL_CONFIG


class AudioClassifier(nn.Module):
    """
    EfficientNet-based audio classifier for multi-label classification.
    """
    def __init__(self, n_classes, pretrained=False):
        super().__init__()
        
        # Load EfficientNet-B0
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool=''
        )
        
        # Get feature dimension
        self.feature_dim = 1280  # EfficientNet-B0 feature dim
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def download_model_from_hf(model_path):
    """Download model from Hugging Face if not present locally"""
    
    # Hugging Face model URL
    HF_MODEL_URL = "https://huggingface.co/sruthisree11/music-instrument-detector-model/resolve/main/model.pt"
    
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create placeholders that we can clear
    info_placeholder = st.empty()
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    info_placeholder.info("🤗 Downloading model from Hugging Face (first time only, ~57MB)...")
    
    try:
        response = requests.get(HF_MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                
                with progress_placeholder:
                    progress_bar = st.progress(0)
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = int((downloaded / total_size) * 100)
                        progress_bar.progress(progress / 100)
                        status_placeholder.text(f"Downloading: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB")
        
        # Clear all download messages
        info_placeholder.empty()
        progress_placeholder.empty()
        status_placeholder.empty()
        
        return True
        
    except Exception as e:
        info_placeholder.empty()
        progress_placeholder.empty()
        status_placeholder.empty()
        st.error(f"❌ Error downloading model: {str(e)}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return False


def load_model(device='cpu'):
    """
    Load trained model.
    Downloads from Hugging Face if not present locally.
    
    Args:
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in eval mode
    """
    model_path = MODEL_CONFIG['model_path']
    
    # Check if model exists, download if not (silently)
    if not os.path.exists(model_path):
        success = download_model_from_hf(model_path)
        if not success:
            raise FileNotFoundError(f"Failed to download model from Hugging Face")
    
    # Load metadata to get number of classes
    with open(MODEL_CONFIG['metadata_path'], 'r') as f:
        metadata = json.load(f)
    
    n_classes = metadata['n_classes']
    
    # Create model
    model = AudioClassifier(n_classes=n_classes, pretrained=False)
    
    # Load weights
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def load_metadata():
    """
    Load model metadata.
    
    Returns:
        Dictionary with metadata
    """
    with open(MODEL_CONFIG['metadata_path'], 'r') as f:
        metadata = json.load(f)
    return metadata


def load_thresholds():
    """
    Load optimized classification thresholds.
    
    Returns:
        Dictionary mapping instrument names to thresholds
    """
    with open(MODEL_CONFIG['thresholds_path'], 'r') as f:
        thresholds = json.load(f)
    return thresholds


def get_device():
    """
    Get the best available device.
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')