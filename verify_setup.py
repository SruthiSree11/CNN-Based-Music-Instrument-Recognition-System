#!/usr/bin/env python3
"""
Setup verification script.
Run this to check if everything is properly configured.
"""

import os
import sys
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT FOUND")
        return False

def check_directory(dirpath, description):
    """Check if a directory exists."""
    if Path(dirpath).exists():
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description}: {dirpath} NOT FOUND")
        return False

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} installed")
        return True
    except ImportError:
        print(f"❌ {module_name} NOT INSTALLED")
        return False

def main():
    print("=" * 60)
    print("🔍 Instrument Detector App - Setup Verification")
    print("=" * 60)
    
    all_good = True
    
    # Check directory structure
    print("\n📁 Checking directory structure...")
    dirs_to_check = [
        ("models", "Models directory"),
        ("utils", "Utils directory"),
        ("configs", "Configs directory"),
        ("assets/sample_audios", "Sample audios directory"),
        (".streamlit", "Streamlit config directory"),
    ]
    
    for dirpath, description in dirs_to_check:
        if not check_directory(dirpath, description):
            all_good = False
    
    # Check critical files
    print("\n📄 Checking critical files...")
    files_to_check = [
        ("app.py", "Main application"),
        ("requirements.txt", "Requirements file"),
        ("README.md", "README"),
        ("configs/app_config.py", "App config"),
        ("utils/__init__.py", "Utils init"),
        ("utils/audio_processing.py", "Audio processing"),
        ("utils/model_loader.py", "Model loader"),
        ("utils/inference.py", "Inference"),
        ("utils/visualization.py", "Visualization"),
        (".streamlit/config.toml", "Streamlit config"),
    ]
    
    for filepath, description in files_to_check:
        if not check_file(filepath, description):
            all_good = False
    
    # Check model files
    print("\n🤖 Checking model files...")
    model_files = [
        ("models/model.pt", "Model weights"),
        ("models/metadata.json", "Model metadata"),
        ("models/thresholds.json", "Thresholds"),
    ]
    
    model_files_exist = True
    for filepath, description in model_files:
        if not check_file(filepath, description):
            model_files_exist = False
            all_good = False
    
    if not model_files_exist:
        print("\n⚠️  MODEL FILES MISSING!")
        print("Please copy these files from your Kaggle notebook:")
        print("  1. /kaggle/working/outputs/best_model.pt → models/model.pt")
        print("  2. /kaggle/working/outputs/metadata.json → models/metadata.json")
        print("  3. /kaggle/working/outputs/thresholds.json → models/thresholds.json")
        print("\nSee models/README.md for detailed instructions.")
    
    # Check dependencies
    print("\n📦 Checking Python dependencies...")
    dependencies = [
        "streamlit",
        "torch",
        "librosa",
        "numpy",
        "matplotlib",
        "plotly",
        "timm",
    ]
    
    deps_installed = True
    for dep in dependencies:
        if not check_import(dep):
            deps_installed = False
            all_good = False
    
    if not deps_installed:
        print("\n⚠️  DEPENDENCIES MISSING!")
        print("Please install them with:")
        print("  pip install -r requirements.txt")
    
    # Check sample audio files (optional)
    print("\n🎵 Checking sample audio files (optional)...")
    sample_dir = Path("assets/sample_audios")
    audio_extensions = ['.ogg', '.wav', '.mp3', '.flac', '.m4a']
    sample_files = []
    
    if sample_dir.exists():
        for ext in audio_extensions:
            sample_files.extend(list(sample_dir.glob(f"*{ext}")))
    
    if sample_files:
        print(f"✅ Found {len(sample_files)} sample audio file(s)")
        for f in sample_files[:5]:  # Show first 5
            print(f"   - {f.name}")
        if len(sample_files) > 5:
            print(f"   ... and {len(sample_files) - 5} more")
    else:
        print("ℹ️  No sample audio files found (this is optional)")
        print("   You can add sample files to assets/sample_audios/")
    
    # Final summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou're ready to run the app:")
        print("  streamlit run app.py")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running the app.")
        print("See SETUP.md for detailed setup instructions.")
    print("=" * 60)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
