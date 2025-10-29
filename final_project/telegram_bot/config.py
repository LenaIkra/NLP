# telegram_bot/config.py

import os
from dotenv import load_dotenv
load_dotenv()

# 🔐 Токен Telegram бота
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")

# 📁 Пути к модулям
AUDIO_PIPELINE_PATH = "../audio_pipeline/audio_pipeline.py"
CLASSIFIER_PATH = "../text_classifier/inference_model.py"

# 📂 Временная директория
TMP_DIR = "../data/"