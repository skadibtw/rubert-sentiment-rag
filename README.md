# Анализ тональности отзывов (Russian Review Sentiment Analysis)

Комплексный (end-to-end) проект в области обработки естественного языка (NLP) для классификации тональности русскоязычных отзывов. 
Проект охватывает полный цикл ML-разработки: от подготовки данных до деплоя модели и создания вопросно-ответной системы (RAG).

## О проекте

Данный проект разработан для демонстрации компетенций в ML-инженерии. Основной фокус сделан на построении готового к интеграции продукта, а не только на обучении моделей в Jupyter Notebook. 

Ключевые особенности и реализованные задачи:
- Обработка данных: очистка текста и оптимизированная загрузка закэшированных датасетов.
- Моделирование: сравнение классического подхода (TF-IDF + Logistic Regression) с современными архитектурами (fine-tuning ruBERT).
- Обучение: реализация кастомного цикла обучения на PyTorch, а также использование Hugging Face Trainer.
- Оценка качества: автоматизированное сохранение метрик, отчетов и графиков для анализа ошибок.
- Инференс: упаковка модели в REST API (FastAPI) для интеграции.
- MLOps: отслеживание экспериментов и параметров с помощью MLflow.
- RAG (Retrieval-Augmented Generation): интеграция векторной базы данных для семантического поиска и ответов на вопросы по корпусу отзывов.
- Инфраструктура: конфигурационные файлы (YAML/JSON), модульные тесты, контейнеризация Docker и пайплайны CI/CD (GitHub Actions).

## Стек технологий

- Machine Learning: Python, PyTorch, Hugging Face Transformers, Datasets, scikit-learn
- MLOps & RAG: MLflow, LangChain, ChromaDB, sentence-transformers, pandas
- Backend & API: FastAPI, Uvicorn
- Инфраструктура и качество: pytest, Ruff, Docker, GitHub Actions

## Результаты оценки моделей

Текущие метрики для обученных моделей:

| Модель | Accuracy | Macro F1 | Примечание |
|-------|----------|----------|-------|
| `bert_custom` | 0.9404 | 0.9404 | Кастомный цикл обучения PyTorch |
| `bert_trainer` | 0.7760 | 0.7787 | Обучение через Hugging Face Trainer |
| `baseline` | 0.7527 | 0.7544 | Базовая модель: TF-IDF + LogisticRegression |

## Структура репозитория

```text
├── configs/                  # YAML-конфигурации для обучения и API
├── scripts/                  # Скрипты запуска конвейеров (точка входа)
├── src/
│   ├── api/                  # FastAPI сервис и веб-интерфейс
│   ├── baselines/            # Базовые классические ML-модели
│   ├── bert/                 # Пайплайны для обучения BERT (Trainer и кастомный)
│   ├── data/                 # Загрузка, обработка и кэширование датасета
│   ├── evaluation/           # Расчет метрик и генерация отчетов
│   ├── inference/            # Оптимизированный инференс батчами
│   ├── models/               # Архитектура классификатора на базе ruBERT
│   ├── rag/                  # Построение векторного индекса и RAG
│   ├── training/             # Кастомная реализация цикла обучения PyTorch
│   └── utils/                # Утилиты и парсинг конфигураций
├── tests/                    # Unit и API тесты
├── Dockerfile                # Конфигурация Docker-образа
└── requirements.txt          # Зависимости проекта
```

## Быстрый старт

### 1. Установка окружения

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

### 2. Подготовка датасета

```bash
python -m src.data.dataset
```

### 3. Обучение моделей

Запуск обучения базовой модели:
```bash
python -m scripts.train_baseline
```

Запуск обучения нейросетевых моделей:
```bash
python -m scripts.train_bert
python -m scripts.train_bert_custom
```

### 4. Оценка и сравнение

Генерация отчетов и сравнение всех сохраненных моделей:
```bash
python -m scripts.compare_models
python -m scripts.export_eval_plots
```

### 5. Запуск RAG и API сервиса

Сборка индексной базы:
```bash
python -m scripts.build_rag_index
```

Запуск API сервера (FastAPI):
```bash
python -m scripts.run_api
```

После запуска:
- Демонстрационный веб-интерфейс доступен по адресу: `http://127.0.0.1:8000/`
- Документация API (Swagger UI) доступна по адресу: `http://127.0.0.1:8000/docs`

## Использование API

### Прогноз тональности (`POST /predict`)
```json
{
  "text": "Приложение стало работать заметно лучше после обновления"
}
```

### Вопрос по базе отзывов (RAG) (`POST /ask`)
```json
{
  "question": "На что чаще всего жалуются после обновления?",
  "top_k": 5,
  "sentiment_focus": "negative",
  "generation_mode": "auto"
}
```

## Docker

Сборка и запуск контейнера:
```bash
docker build -t nlp-pet-project .
docker run --rm -p 8000:8000 -e MODEL_DIR=artifacts/baseline nlp-pet-project
```

## Дополнительно: RAG и LLM

Система RAG работает локально "из коробки" (извлекая ответы напрямую из индекса). Для генерации финального текста с помощью LLM (например, OpenAI или совместимых локальных серверов) достаточно передать параметры через переменные окружения: `OPENAI_API_KEY` и `OPENAI_BASE_URL`.
