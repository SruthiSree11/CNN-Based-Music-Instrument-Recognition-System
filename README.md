# 🎵 Music Instrument Detector

AI-powered temporal instrument classification using deep learning (EfficientNet-B0).

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**Login:** `demo` / `demo123`

---

## 📁 Project Structure

```
instrument_detector_app/
├── app.py                    # Main Streamlit application
├── auth.py                   # Authentication system
├── requirements.txt          # Python dependencies
├── users.json               # User credentials (local only)
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── configs/
│   └── app_config.py        # App settings
├── models/
│   ├── model.pt             # EfficientNet-B0 weights
│   ├── metadata.json        # Model metadata
│   └── thresholds.json      # Detection thresholds
└── utils/
    ├── __init__.py
    ├── audio_processing.py  # Audio loading/preprocessing
    ├── model_loader.py      # Model initialization
    ├── inference.py         # Prediction logic
    └── visualization.py     # Plotly visualizations
```

---

## 🎯 Features

### Detection
- ✅ Temporal analysis (sliding window)
- ✅ 20 instrument classes
- ✅ Configurable sensitivity
- ✅ Multiple aggregation strategies (max, mean, vote)

### Visualization
- ✅ Confidence scores bar chart
- ✅ Audio waveform
- ✅ Mel spectrogram
- ✅ Temporal timeline
- ✅ Confidence heatmap

### Export
- ✅ JSON download (machine-readable)
- ✅ PDF report with plots
- ✅ Temporal timeline table

### Security
- ✅ SHA-256 password hashing
- ✅ Session-based authentication
- ✅ Logout functionality

---


### Add New Users

Edit `users.json` or use the auth system to add users programmatically.

---

## 🔧 Configuration

### Supported Audio Formats
- OGG, WAV, MP3, FLAC, M4A

### Model Details
- **Architecture:** EfficientNet-B0
- **Performance:** 65.2% Macro F1
- **Dataset:** OpenMIC-2018
- **Classes:** 20 instruments

### Detection Settings
- **Sensitivity:** 0.5 (more detections) to 2.0 (stricter)
- **Aggregation:** max, mean, or vote
- **Window:** Sliding window temporal analysis

---

## 📊 Supported Instruments

1. accordion
2. banjo
3. bass
4. cello
5. clarinet
6. cymbals
7. drums
8. flute
9. guitar
10. mallet_percussion
11. mandolin
12. organ
13. piano
14. saxophone
15. synthesizer
16. trombone
17. trumpet
18. ukulele
19. violin
20. voice

---

## 🐛 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### Model file missing
Ensure `models/model.pt` exists (150MB file)

### PDF generation slow
This is normal for first generation. Subsequent generations are cached.

### Audio upload fails
- Check file format (must be: ogg, wav, mp3, flac, m4a)
- Ensure file size < 200MB

---

## 📝 Requirements

```
streamlit>=1.32.0
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.14.0
timm>=0.9.0
soundfile>=0.12.0
reportlab>=4.0.0
kaleido>=0.2.1
```

---

## 🔐 Security Notes

- **Local Development:** Uses `users.json` with SHA-256 hashed passwords
- **Production:** Consider using Streamlit Secrets or environment variables
- **Default Credentials:** Change `demo/demo123` before deployment

---

## 📈 Performance Tips

1. **First Load:** Model loading takes ~5s (cached after first run)
2. **Analysis:** Temporal analysis takes ~3-10s depending on audio length
3. **PDF Generation:** First generation ~5s, subsequent ~1s (cached plots)

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project uses the OpenMIC-2018 dataset and EfficientNet-B0 architecture.

---

## 🆘 Support

For issues or questions:
1. Check troubleshooting section above
2. Review Streamlit documentation
3. Open an issue on GitHub

---

## 🎓 Credits

- **Model:** EfficientNet-B0 (PyTorch/timm)
- **Dataset:** OpenMIC-2018
- **Framework:** Streamlit
- **Visualizations:** Plotly

---

**Version:** 1.0.0  
**Last Updated:** January 2026  
**Status:** Production Ready ✅