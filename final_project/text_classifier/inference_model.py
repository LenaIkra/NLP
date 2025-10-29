# ================================
# 🧠 Text Classification Inference
# ================================

import pickle
import os
import numpy as np

# --- Пути ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "svm_best_model.pkl")

# --- Загрузка модели ---
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --- Маппинг числовых меток ---
# (если LabelEncoder не сохранён — задаём вручную)
LABELS = {
    2: "low",
    1: "medium",
    0: "high"
}

def predict_priority(text: str) -> str:
    """Возвращает предсказанный приоритет текста (low/medium/high)."""
    if not text or len(text.strip()) == 0:
        return "unknown"

    try:
        pred = model.predict([text])[0]
    except Exception as e:
        print("⚠️ Prediction error:", e)
        return "unknown"

    # Если возвращено число → конвертируем
    if isinstance(pred, (np.integer, int, float)):
        pred = int(pred)
        return LABELS.get(pred, f"unknown ({pred})")

    # Если уже строка
    return str(pred).lower()