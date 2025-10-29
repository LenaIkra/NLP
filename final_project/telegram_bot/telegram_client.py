import os
import requests
from pydub import AudioSegment
import telebot
import whisper
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt

from telegram_bot.config import BOT_TOKEN
from text_classifier.inference_model import predict_priority  # 🧠 классификатор





# === инициализация ===
bot = telebot.TeleBot(BOT_TOKEN)
model = whisper.load_model("small.en")
os.makedirs("data", exist_ok=True)

# === шумоподавление и фильтрация ===
def denoise_and_filter(wav_path, out_path):
    """Удаляет шумы и лишние частоты из аудио"""
    y, sr = librosa.load(wav_path, sr=16000)
    print(f"🔊 Loaded {wav_path}, duration {len(y)/sr:.2f}s")

    # 1️⃣ spectral gating
    noise_len = int(0.5 * sr)
    noise_clip = y[:noise_len] if len(y) > noise_len else y
    y_nr = nr.reduce_noise(y=y, y_noise=noise_clip, sr=sr)

    # 2️⃣ band-pass 80–7000 Hz
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        return b, a

    b, a = butter_bandpass(80, 7000, sr, order=5)
    y_bp = filtfilt(b, a, y_nr)

    sf.write(out_path, y_bp, sr)
    print(f"✅ Cleaned audio saved to {out_path}")
    return out_path


# === команды ===
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(
        message,
        "👋 Hi! Send me a *voice message* in English — I'll clean it, transcribe it, classify it by priority, and send the result back.",
        parse_mode="Markdown"
    )


# === обработка голосовых сообщений ===
@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    try:
        print("🎧 Voice message received...")

        # === 1️⃣ Скачиваем .ogg ===
        file_info = bot.get_file(message.voice.file_id)
        file_path = file_info.file_path
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"

        ogg_path = "data/input.ogg"
        wav_path = "data/input.wav"
        clean_path = "data/cleaned.wav"

        r = requests.get(file_url)
        with open(ogg_path, "wb") as f:
            f.write(r.content)
        print("✅ File downloaded")

        # === 2️⃣ Конвертируем OGG → WAV ===
        audio = AudioSegment.from_file(ogg_path, format="ogg")
        audio.export(wav_path, format="wav")
        print("🔄 Converted to WAV")

        # === 3️⃣ Предобработка ===
        denoise_and_filter(wav_path, clean_path)

        # === 4️⃣ Распознавание речи (Whisper) ===
        result = model.transcribe(clean_path)
        text = result.get("text", "").strip()
        print("🧠 Recognized text:", text)

        if not text:
            bot.reply_to(message, "⚠️ Couldn’t recognize any speech.")
            return

        # === 5️⃣ Классификация приоритета ===
        priority = predict_priority(text).lower()

        emoji = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🔴",
            "unknown": "⚪️"
        }.get(priority, "⚪️")

        print(f"🎯 Predicted priority: {priority.upper()}")

        # === 6️⃣ Ответ пользователю ===
        reply = (
            f"🗣️ *Recognized text:*\n{text}\n\n"
            f"{emoji} *Predicted priority:* {priority.upper()}"
        )
        bot.reply_to(message, reply, parse_mode="Markdown")

    except Exception as e:
        print("❌ Error:", e)
        bot.reply_to(message, f"⚠️ Error while processing audio: {e}")


print("🤖 Bot is running with noise reduction + Whisper + SVM classification...")
bot.infinity_polling()