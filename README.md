# Russian Review Sentiment Analysis

An end-to-end NLP pet project for sentiment classification of Russian app reviews.
The repository includes dataset preparation, a TF-IDF baseline, two ruBERT training pipelines,
evaluation artifacts, inference utilities, a FastAPI service, and a lightweight browser demo.

## Why this project exists

This project is built to show practical ML engineering skills, not just model fine-tuning in a notebook:

- data cleaning and cached dataset loading
- classical baseline vs transformer comparison
- custom PyTorch training loop alongside Hugging Face Trainer
- repeatable evaluation with saved reports and error analysis
- inference packaging for API usage
- project ergonomics: configs, tests, Docker, and CI

## Stack

- Python, PyTorch, Hugging Face Transformers, Datasets
- scikit-learn, pandas, FastAPI, Uvicorn
- pytest, Ruff, GitHub Actions

## Repository layout

```text
nlp_pet_project/
├── configs/                  # YAML configs for training, comparison, and API
├── scripts/                  # entrypoints that read configs and launch modules
├── src/
│   ├── api/                  # FastAPI app and browser demo
│   ├── baselines/            # TF-IDF + LogisticRegression baseline
│   ├── bert/                 # Trainer and custom-loop training pipelines
│   ├── data/                 # dataset loading and text cleaning
│   ├── evaluation/           # metrics, reports, comparison helpers
│   ├── inference/            # predictor loading and batch inference
│   ├── models/               # ruBERT classifier module
│   ├── training/             # custom trainer implementation
│   └── utils/                # shared config helpers
├── tests/                    # unit and API smoke tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## Results

Current local runs cover the baseline and both transformer variants:

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| `bert_trainer` | 0.7830 | 0.7848 | Best result so far |
| `bert_custom` | 0.7760 | 0.7787 | Custom PyTorch loop |
| `baseline` | 0.7452 | 0.7468 | TF-IDF + LogisticRegression |

The project also saves:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.csv`
- `errors.csv`

for each trained model under `artifacts/`.

## Quick start

### 1. Create environment

```bash
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements-dev.txt
```

If you already have a working virtual environment, reuse it.
If you are on ROCm or another custom PyTorch build, install `torch` your own way first and then install the rest of the dependencies.

### 2. Prepare dataset cache

```bash
python -m src.data.dataset
```

### 3. Run baseline

```bash
python -m scripts.train_baseline
```

### 4. Train models

```bash
python -m scripts.train_bert
python -m scripts.train_bert_custom
```

### 5. Compare all saved runs

```bash
python -m scripts.compare_models
```

### 6. Launch API and demo

```bash
python -m scripts.run_api
```

Open:

- `http://127.0.0.1:8000/` for the browser demo
- `http://127.0.0.1:8000/docs` for Swagger UI

## Config-driven workflow

All main entrypoints read YAML configs from `configs/`.

- `configs/train_baseline.yaml`
- `configs/train_bert.yaml`
- `configs/train_bert_custom.yaml`
- `configs/compare_models.yaml`
- `configs/api.yaml`

You can keep defaults in config files and still override them by editing YAML before launch.

## API

### `GET /health`

Returns service status and whether a local model directory is available.

### `POST /predict`

Request:

```json
{
  "text": "Приложение стало работать заметно лучше после обновления"
}
```

Response:

```json
{
  "text": "Приложение стало работать заметно лучше после обновления",
  "label_id": 2,
  "label": "positive",
  "scores": {
    "negative": 0.012345,
    "neutral": 0.107654,
    "positive": 0.879999
  }
}
```

## Demo UI

The root page serves a small interactive interface for manual testing.
It uses the same `/predict` endpoint as the API, so the UI and backend stay aligned.

## Testing and linting

```bash
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pytest
```

## Docker

Build and run:

```bash
docker build -t nlp-pet-project .
docker run --rm -p 8000:8000 -e MODEL_DIR=artifacts/bert_custom nlp-pet-project
```

The container expects model files to be available inside the image or mounted at runtime.

## Notes on artifacts

- Dataset cache under `data/cache/` is ignored by git.
- Model checkpoints and evaluation artifacts under `artifacts/` are ignored by git.
- The repository tracks code, configs, and reproducible instructions instead of large binaries.

## Next ideas

- add a RAG module over the review corpus
- export plots for confusion matrix and learning curves in README
- add experiment tracking with W&B or MLflow
