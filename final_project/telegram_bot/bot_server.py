"""
bot_server.py — Flask-сервер для Telegram-бота (вариант для деплоя)

📋 Назначение:
Этот модуль реализует серверную часть Telegram-бота с помощью Flask.
В отличие от telegram_client.py (который работает через polling),
здесь используется webhook — Telegram сам отправляет запросы на сервер,
а бот отвечает пользователю через API.

🧠 Применение:
- Используется при развёртывании проекта на хостинге (например, Render, Railway, PythonAnywhere).
- Подходит для деплоя Docker-контейнера с HTTPS.

⚙️ Локально использовать НЕ нужно.
Для локальных тестов — запускайте telegram_client.py
"""

import os
from flask import Flask, request
import telebot
from telegram_bot.config import BOT_TOKEN

# === Flask app initialization ===
app = Flask(__name__)
bot = telebot.TeleBot(BOT_TOKEN)

@app.route("/", methods=["GET"])
def index():
    """Проверка, что сервер запущен"""
    return "🤖 Telegram Bot Server is running!", 200

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    """Получение обновлений от Telegram"""
    json_str = request.get_data().decode("UTF-8")
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return "ok", 200


if __name__ == "__main__":
    # --- Настройки для локального запуска ---
    PORT = int(os.environ.get("PORT", 5000))
    print(f"🌐 Starting Flask bot server on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT)