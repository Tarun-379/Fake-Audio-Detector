🎙️ Deepfake Audio Detector

Detect whether an audio clip is real human speech or AI-generated (deepfake) using deep learning.

🚀 Overview

With the rise of AI voice cloning tools, generating highly realistic fake audio has become easy. This project builds a robust deep learning system that can distinguish between:

✅ Real human voice (bonafide)
❌ AI-generated / cloned voice (spoof)

The model achieves AUC = 1.0 on the ASVspoof 2019 dataset.

🎯 Problem Statement

AI-generated voices can be used for:

Fraud and scams
Impersonation
Misinformation

This project answers:

“Is this voice real or AI-generated?”

📊 Dataset
Name: ASVspoof 2019 (Logical Access)
Size: ~25GB
Audio: .flac, 16kHz mono
Classes:
bonafide → Real
spoof → Fake

⚠️ Dataset is highly imbalanced (~9:1 fake vs real)

🧠 Model Architecture
🔹 Input Pipeline

Audio → 3-channel spectrogram (128×128)

Channels:

Mel Spectrogram
MFCC
CQT
🔹 Model

EfficientNet-B0 + Bidirectional LSTM

Input (3×128×128)
      ↓
EfficientNet-B0 (Feature Extraction)
      ↓
Feature Vector (1280)
      ↓
BiLSTM (Temporal Learning)
      ↓
Fully Connected Layers
      ↓
Output: Real / Fake
⚙️ Training Details
Parameter	Value
Optimizer	AdamW
Learning Rate	1e-4
Loss Function	Focal Loss
Batch Size	32
Epochs	15
💡 Why Focal Loss?

Handles class imbalance by focusing on hard samples.

📈 Results
Metric	Score
AUC	1.0000
Accuracy	~99%
EER	~0%
🛠️ Tech Stack
Deep Learning: PyTorch, timm
Audio Processing: librosa, soundfile
Backend: Flask
Frontend: HTML, CSS, JavaScript
Deployment: Kaggle (T4 GPU)
🌐 Web Application
Features
🎧 Upload audio files (MP3, WAV, FLAC, OGG)
🎤 Record audio directly in browser
📊 Live waveform visualization
⚡ Instant prediction
📈 Confidence score display
🔄 Workflow
User Input (Upload / Record)
        ↓
Flask API
        ↓
Feature Extraction (Mel + MFCC + CQT)
        ↓
Model Prediction
        ↓
Result Display (UI)
🔥 Key Innovations
✅ 3-channel spectrogram (treat audio as image)
✅ Transfer learning from ImageNet
✅ CNN + RNN hybrid model
✅ Focal loss for imbalance
✅ Feature caching for faster training
⚠️ Limitations
Works best on known spoofing systems
Performance may drop on unseen TTS models
Generalization is still an open problem
🔮 Future Work
Grad-CAM visualization
Add ASVspoof 2021 dataset
Convert to ONNX / TFLite
Real-time streaming detection
Deploy on cloud (Hugging Face / AWS)
📦 Installation
git clone https://github.com/your-username/deepfake-audio-detector.git
cd deepfake-audio-detector

pip install -r requirements.txt
▶️ Usage
python app.py

Then open:

http://localhost:5000
📁 Project Structure
├── model/
├── data/
├── features/
├── app.py
├── utils.py
├── requirements.txt
└── README.md
💬 FAQ

Q: Why not just use accuracy?
A: Accuracy is misleading for imbalanced data. AUC is more reliable.

Q: Can it detect all AI voices?
A: Not perfectly. Performance drops on unseen models.

📜 License

This project is for academic and research purposes.

👨‍💻 Author

Adil
Electronics and Communication Engineering

⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!
