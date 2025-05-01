### README.md
# 🎬 PySpark Movie Recommender System

## 📌 Описание
Проект создаёт рекомендательную систему на основе алгоритма ALS из PySpark MLlib, используя датасет MovieLens 20M, автоматически загружаемый через KaggleHub.

## 📁 Структура
- `src/` — код проекта
- `output/` — предсказания и метрики

## 🚀 Запуск проекта
```bash
pip install -r requirements.txt
python main.py
```

## 📤 Результаты
- `output/recommendations.ыйд` — таблица рекомендаций (userId, movieId, score, title)
- `output/rmse.txt` — RMSE ошибки модели
