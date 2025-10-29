# ======================================================
# 🟩 AUDIO → TEXT PIPELINE (Noise reduction + WhisperX)
# ======================================================

import librosa, noisereduce as nr, numpy as np, soundfile as sf
from scipy.signal import butter, filtfilt
import whisperx
import torch
import os

# ======================================================
# 1. UPLOAD AUDIO
# ======================================================
# === 📂 Local file selection (for VS Code / local run) ===
audio_path = "data/Low priority.m4a"

if not os.path.exists(audio_path):
    raise FileNotFoundError(f"❌ Audio file not found: {audio_path}")
else:
    print(f"✅ Using local audio file: {audio_path}")

# ======================================================
# 2. LOAD + NOISE REDUCTION + BAND-PASS FILTER
# ======================================================
print("🎧 Loading and cleaning audio...")

y, sr = librosa.load(audio_path, sr=16000, mono=True)

# --- Noise reduction (spectral gating) ---
noise_len = int(0.5 * sr)
noise_clip = y[:noise_len] if len(y) > noise_len else y
y_nr = nr.reduce_noise(y=y, y_noise=noise_clip, sr=sr)

# --- Band-pass filter (speech band: 80–7000 Hz) ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

b, a = butter_bandpass(80, 7000, sr, order=5)
y_bp = filtfilt(b, a, y_nr)

# --- Save cleaned audio ---
os.makedirs("data", exist_ok=True)
clean_path = "data/clean.wav"
sf.write(clean_path, y_bp, sr)
print(f"✅ Cleaned audio saved to: {clean_path}")

# ======================================================
# 3. TRANSCRIBE WITH WHISPERX
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Using device: {device}")

try:
    model = whisperx.load_model("small.en", device=device, compute_type="float32")
except Exception as e:
    print("⚠️ WhisperX failed to load, trying openai-whisper fallback:", e)
    import whisper
    model = whisper.load_model("small.en", device=device)

print("🎧 Transcribing...")

# --- load audio and transcribe ---
audio = whisperx.load_audio(clean_path)
result = model.transcribe(audio)

print("\n🗣️ Recognized text:\n")

# --- handle output format variations ---
if isinstance(result, dict):
    if "text" in result:
        text_output = result["text"]
    elif "segments" in result:
        text_output = " ".join([seg.get("text", "") for seg in result["segments"]])
    else:
        text_output = str(result)
else:
    text_output = str(result)

print(text_output.strip())

# --- save to file ---
output_txt = "data/recognized_text.txt"
with open(output_txt, "w") as f:
    f.write(text_output.strip())

print(f"\n💾 Transcription saved to: {output_txt}")