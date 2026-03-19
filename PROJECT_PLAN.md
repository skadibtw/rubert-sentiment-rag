# NLP Pet Project: Sentiment Analysis + RAG

## Цель проекта

Fine-tuning ruBERT для классификации тональности русскоязычных отзывов + RAG-система для вопросов по корпусу отзывов. Стек: PyTorch, HuggingFace Transformers, LangChain, FAISS, FastAPI.

---

## Что этот проект демонстрирует работодателю

- Понимание архитектуры трансформеров и transfer learning
- Умение писать training loop на PyTorch (а не только `Trainer.train()`)
- Работа с данными: очистка, токенизация, разбиение
- Оценка модели по адекватным метрикам (не только accuracy)
- Практическое применение LLM через RAG
- Умение оформить проект: структура, конфиги, воспроизводимость

---

## Часть 1: Fine-tuning ruBERT для Sentiment Analysis

### 1.1 Выбор модели

**Рекомендуемая модель**: `ai-forever/ruBERT-base`

| Модель | Параметры | VRAM | Ссылка |
|--------|-----------|------|--------|
| `ai-forever/ruBERT-base` | 178M | ~2GB | <https://huggingface.co/ai-forever/ruBERT-base> |
| `ai-forever/ruBERT-large` | 427M | ~4GB | <https://huggingface.co/ai-forever/ruBERT-large> |
| `cointegrated/rubert-tiny2` | 29M | <1GB | <https://huggingface.co/cointegrated/rubert-tiny2> |
| `DeepPavlov/rubert-base-cased` | 178M | ~2GB | <https://huggingface.co/DeepPavlov/rubert-base-cased> |

`ruBERT-base` — оптимальный баланс качества и размера. Помещается в любой GPU с 4GB+ VRAM, в том числе бесплатный Colab (T4).

### 1.2 Датасеты

**Основной вариант**: `RuReviews` — отзывы на приложения из Google Play, 3 класса (positive/neutral/negative), ~90k примеров.

| Датасет | Классы | Размер | Ссылка |
|---------|--------|--------|--------|
| RuReviews | 3 (pos/neut/neg) | ~90k | <https://github.com/sismetanern/rureviews> |
| RuSentiment | 5 классов | ~31k | <https://github.com/text-machine-lab/rusentiment> |
| SentiRuEval | 4 класса | ~12k (Twitter) | <https://github.com/mokoron/sentirueval> |
| IMDB на русском (переводы) | 2 (pos/neg) | ~50k | можно через `datasets` library |
| Собственный датасет (парсинг) | на выбор | на выбор | Otzovik, IRecommend, Wildberries |

Если хочешь выделиться — собери свой датасет парсингом (покажет умение работать с «грязными» данными). Но это опционально, RuReviews достаточно.

### 1.3 Подготовка данных

Шаги:

1. Загрузка датасета (через HuggingFace `datasets` или pandas)
2. Очистка текста: удаление HTML, лишних пробелов, нормализация
3. Exploratory Data Analysis: распределение классов, длины текстов, примеры
4. Разбиение: train/val/test (80/10/10) со стратификацией по классам
5. Токенизация через `AutoTokenizer` с padding и truncation (max_length=512)
6. Создание `torch.utils.data.Dataset` и `DataLoader`

**Важно**: написать свой Dataset-класс, а не использовать только `datasets.map()` — это покажет понимание PyTorch.

### 1.4 Training Pipeline

**Рекомендация**: написать кастомный training loop на PyTorch. Использовать `Trainer` из HuggingFace можно для сравнения, но кастомный loop — ключевой навык, который оценят на собеседовании.

Что должно быть в training loop:

- Перебор батчей с `DataLoader`
- Forward pass, loss calculation (`CrossEntropyLoss`)
- Backward pass, `optimizer.step()`, `scheduler.step()`
- Gradient clipping (`torch.nn.utils.clip_grad_norm_`)
- Валидация каждые N шагов/эпох
- Логирование метрик (в W&B или хотя бы в консоль)
- Сохранение лучшей модели по val loss/f1
- Early stopping (опционально)

Гиперпараметры для старта:

```
learning_rate: 2e-5
batch_size: 16-32
epochs: 3-5
warmup_steps: 10% от общего числа шагов
weight_decay: 0.01
optimizer: AdamW
scheduler: linear с warmup
max_seq_length: 256-512
```

### 1.5 Evaluation

Метрики:

- **Accuracy** — базовая, но недостаточная при дисбалансе классов
- **F1-score (macro и weighted)** — основная метрика для отчёта
- **Precision / Recall по каждому классу**
- **Confusion Matrix** — визуализация ошибок
- **Classification Report** через `sklearn.metrics.classification_report`

Что ещё стоит сделать:

- Сравнить с бейзлайном (TF-IDF + LogisticRegression) — показывает, что трансформер реально даёт прирост
- Анализ ошибок: примеры, где модель ошибается, и почему
- Визуализация: графики loss/f1 по эпохам, confusion matrix heatmap

### 1.6 Логирование экспериментов

Рекомендация: **Weights & Biases (wandb)** — бесплатный для персональных проектов, красивые дашборды, хорошо смотрится в резюме.

Альтернатива: **MLflow** (локальный, без регистрации).

Что логировать:

- Гиперпараметры
- Train/val loss и метрики по шагам
- Confusion matrix как артефакт
- Лучший чекпоинт модели

Ссылки:

- <https://docs.wandb.ai/quickstart>
- <https://mlflow.org/docs/latest/quickstart.html>

---

## Часть 2: RAG Pipeline

### 2.1 Идея

Построить систему, которая отвечает на вопросы по корпусу отзывов. Например:

- «На что чаще всего жалуются?»
- «Какие плюсы отмечают пользователи?»
- «Есть ли отзывы про скорость работы?»

Это покажет умение работать с LLM, embeddings и vector stores.

### 2.2 Архитектура RAG

```
Вопрос пользователя
        |
        v
  Embedding модель (sentence-transformers)
        |
        v
  Поиск в FAISS (top-k похожих отзывов)
        |
        v
  Формирование промпта: вопрос + найденные отзывы
        |
        v
  LLM генерирует ответ
```

### 2.3 Компоненты

**Embedding модель** (для векторизации отзывов и запросов):

| Модель | Язык | Ссылка |
|--------|------|--------|
| `intfloat/multilingual-e5-base` | multi | <https://huggingface.co/intfloat/multilingual-e5-base> |
| `cointegrated/LaBSE-en-ru` | en+ru | <https://huggingface.co/cointegrated/LaBSE-en-ru> |
| `ai-forever/sbert_large_nlu_ru` | ru | <https://huggingface.co/ai-forever/sbert_large_nlu_ru> |

**Vector Store**: FAISS (простой, локальный, без серверов)

- <https://github.com/facebookresearch/faiss>
- Интеграция через LangChain: `langchain_community.vectorstores.FAISS`

**LLM для генерации**: варианты

- Через API: OpenAI GPT-4o-mini, GigaChat API (бесплатно для разработчиков)
- Локально: `IlyaGusev/saiga_llama3_8b` (если есть GPU) — <https://huggingface.co/IlyaGusev/saiga_llama3_8b>
- Через Ollama: можно запустить llama3/mistral локально

Для пет-проекта достаточно API (дешевле и проще). Но если хочешь показать работу с локальными моделями — Ollama + LangChain.

### 2.4 Реализация через LangChain

Ключевые компоненты LangChain для этой задачи:

- `RecursiveCharacterTextSplitter` — разбивка текстов на чанки
- `HuggingFaceEmbeddings` — обёртка над sentence-transformers
- `FAISS` — vector store
- `RetrievalQA` или `create_retrieval_chain` — готовая RAG-цепочка
- `ChatPromptTemplate` — шаблоны промптов

Документация:

- <https://python.langchain.com/docs/tutorials/rag/>
- <https://python.langchain.com/docs/how_to/vectorstores/>
- <https://python.langchain.com/docs/integrations/vectorstores/faiss/>

### 2.5 Связка с fine-tuned моделью

Интересная идея для проекта: перед тем как отвечать на вопрос через RAG, прогнать найденные отзывы через fine-tuned sentiment classifier. Это позволяет:

- Фильтровать отзывы по тональности
- Добавлять в промпт информацию о тональности
- Пример: «Найди негативные отзывы про доставку» — сначала retrieval, потом фильтр по sentiment

---

## Часть 3: API

### 3.1 FastAPI

Два эндпоинта:

- `POST /predict` — классификация тональности одного текста
- `POST /ask` — вопрос к RAG-системе

Документация:

- <https://fastapi.tiangolo.com/tutorial/>
- Swagger UI генерируется автоматически на `/docs`

---

## Структура проекта

```
nlp_pet_project/
├── configs/
│   └── train_config.yaml          # Гиперпараметры, пути
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py             # Dataset класс, загрузка, очистка
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py          # nn.Module обёртка над ruBERT
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py             # Кастомный training loop
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py             # Метрики, confusion matrix, отчёты
│   ├── rag/
│   │   ├── __init__.py
│   │   └── pipeline.py            # RAG: индексация + retrieval + генерация
│   └── api/
│       ├── __init__.py
│       └── app.py                 # FastAPI
├── scripts/
│   ├── train.py                   # Точка входа: обучение
│   ├── evaluate.py                # Точка входа: оценка на тесте
│   └── run_rag.py                 # Точка входа: RAG demo
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis
├── requirements.txt
├── .gitignore
└── README.md                      # Описание для GitHub
```

---

## Порядок реализации

### Этап 1: Данные (1-2 дня)

1. Скачать RuReviews, загрузить в pandas
2. EDA в jupyter notebook: распределение, длины, примеры
3. Очистка и препроцессинг
4. Написать `torch.utils.data.Dataset` класс
5. Сделать DataLoader с правильным padding (через `DataCollatorWithPadding`)

### Этап 2: Модель и обучение (2-3 дня)

1. Написать класс модели (`nn.Module`, внутри `AutoModel` + classification head)
2. Написать training loop с нуля на PyTorch
3. Добавить валидацию, логирование, сохранение чекпоинтов
4. Обучить, подобрать гиперпараметры
5. Сравнить с бейзлайном (TF-IDF + LogReg)

### Этап 3: Evaluation (1 день)

1. Прогнать на тестовой выборке
2. Classification report, confusion matrix
3. Анализ ошибок: собрать примеры неправильных предсказаний
4. Визуализации для README

### Этап 4: RAG (2-3 дня)

1. Проиндексировать отзывы в FAISS через sentence-transformers
2. Настроить LangChain pipeline (retriever + LLM)
3. Связать с sentiment classifier
4. Написать промпты, протестировать

### Этап 5: API и оформление (1-2 дня)

1. FastAPI с двумя эндпоинтами
2. README.md для GitHub (описание, результаты, как запустить)
3. requirements.txt, .gitignore
4. Опционально: Dockerfile

**Итого: ~7-11 дней** при 3-4 часах в день.

---

## Ключевые библиотеки и документация

### PyTorch

- Туториал по fine-tuning: <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>
- DataLoader: <https://pytorch.org/docs/stable/data.html>
- Optimizers: <https://pytorch.org/docs/stable/optim.html>

### HuggingFace Transformers

- Документация: <https://huggingface.co/docs/transformers/>
- Fine-tuning tutorial: <https://huggingface.co/docs/transformers/training>
- AutoModel: <https://huggingface.co/docs/transformers/model_doc/auto>

### HuggingFace Datasets

- Quickstart: <https://huggingface.co/docs/datasets/quickstart>

### LangChain

- RAG tutorial: <https://python.langchain.com/docs/tutorials/rag/>
- FAISS интеграция: <https://python.langchain.com/docs/integrations/vectorstores/faiss/>
- Документация: <https://python.langchain.com/docs/introduction/>

### Sentence Transformers

- Документация: <https://www.sbert.net/>
- Русские модели: <https://huggingface.co/ai-forever>

### FastAPI

- Tutorial: <https://fastapi.tiangolo.com/tutorial/>

### Weights & Biases

- Quickstart: <https://docs.wandb.ai/quickstart>
- PyTorch интеграция: <https://docs.wandb.ai/guides/integrations/pytorch>

### scikit-learn (метрики)

- classification_report: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html>
- confusion_matrix: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html>

---

## Полезные статьи и туториалы

- **BERT fine-tuning explained**: <https://mccormickml.com/2019/07/22/BERT-fine-tuning/>
- **Illustrated Transformer**: <https://jalammar.github.io/illustrated-transformer/>
- **RAG explained (LangChain blog)**: <https://blog.langchain.dev/tutorial-chatgpt-over-your-data/>
- **DeepPavlov ruBERT**: <http://docs.deeppavlov.ai/en/master/features/models/bert.html>
- **Practical NLP (O'Reilly)**: <https://github.com/practical-nlp/practical-nlp-code>

---

## Советы для резюме

1. **Оформи README с результатами**: таблица метрик, скриншоты confusion matrix, примеры работы RAG
2. **Покажи сравнение**: baseline vs fine-tuned модель (числа)
3. **Опиши, что ты узнал**: какие проблемы возникли и как решил
4. **Код должен быть чистым**: docstrings, type hints, разбиение по модулям
5. **Конфиги отдельно от кода**: yaml/json, а не хардкод
6. **Не коммить данные и модели в git**: используй .gitignore, указывай в README как скачать

---

## Что можно добавить потом (бонусы)

- **Docker** — контейнеризация для деплоя
- **DVC** — версионирование данных и моделей (<https://dvc.org/>)
- **GitHub Actions** — CI для линтинга и тестов
- **Streamlit/Gradio** — простой веб-интерфейс вместо/вместе с FastAPI
- **ONNX export** — оптимизация инференса
- **LoRA fine-tuning** — если захочешь добавить работу с LLM в обучении

---

## Заметки по AMD GPU (RX 9070 XT)

PyTorch поддерживает AMD через ROCm, но:

- RDNA 4 (RX 9070 XT) пока имеет ограниченную поддержку в ROCm
- Для этого проекта проще использовать Google Colab (бесплатный T4) или арендовать GPU (Vast.ai, RunPod — от $0.2/час за RTX 3090)
- ruBERT-base обучается за 20-40 минут на T4, так что аренда обойдётся в копейки
- Следи за обновлениями ROCm: <https://rocm.docs.amd.com/en/latest/>
