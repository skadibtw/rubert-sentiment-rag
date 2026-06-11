# Russian Review Sentiment Analysis

An end-to-end NLP pet project for sentiment classification of Russian app reviews.
The repository includes dataset preparation, a TF-IDF baseline, two ruBERT training pipelines,
evaluation artifacts, inference utilities, a LangChain + ChromaDB RAG layer over the review corpus,
a FastAPI service, and a browser demo.

## Why this project exists

This project is built to show practical ML engineering skills, not just model fine-tuning in a notebook:

- data cleaning and cached dataset loading
- classical baseline vs transformer comparison
- custom PyTorch training loop alongside Hugging Face Trainer
- repeatable evaluation with saved reports and error analysis
- inference packaging for API usage
- optional MLflow experiment tracking
- retrieval-augmented question answering over real review texts
- project ergonomics: configs, tests, Docker, and CI

## Stack

- Python, PyTorch, Hugging Face Transformers, Datasets
- LangChain, ChromaDB, sentence-transformers, scikit-learn, MLflow, pandas, FastAPI, Uvicorn
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
│   ├── rag/                  # local review indexing and retrieval QA
│   ├── training/             # custom trainer implementation
│   └── utils/                # shared config helpers
├── tests/                    # unit and API smoke tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## Results

Current local runs cover the classical baseline and previous full ruBERT
fine-tuning runs:

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| `baseline_binary` | 0.9404 | 0.9404 | Binary negative-vs-positive word+char TF-IDF + LogisticRegression |
| `bert_trainer` | 0.7830 | 0.7848 | Previous full local run; best result so far |
| `bert_custom` | 0.7760 | 0.7787 | Previous custom PyTorch loop run |
| `baseline` | 0.7527 | 0.7544 | Word+char TF-IDF + LogisticRegression |

`baseline_binary` is a separate polarity benchmark that drops neutral reviews
and evaluates only negative-vs-positive classification.

The best full local ruBERT run improves over the classical TF-IDF baseline by
roughly +3.0 macro F1 points.

The project also saves:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.csv`
- `errors.csv`
- `model.joblib` for the scikit-learn baseline
- PNG plots under `artifacts/figures/`

for each trained model under `artifacts/`.

## Quick start

### 1. Create environment

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

If you already have a working virtual environment, reuse it.
If you need a custom PyTorch build, install `torch` your own way first and then install the rest of the dependencies.
For this project, keep `transformers<5`.

### 2. Prepare dataset cache

```bash
python -m src.data.dataset
```

### 3. Run baseline

```bash
python -m scripts.train_baseline
python -m scripts.train_binary_baseline
```

### 4. Train models

```bash
python -m scripts.train_bert
python -m scripts.train_bert_custom
```

For a slow but stable CPU sanity check of the Hugging Face Trainer pipeline:

```bash
python -m scripts.train_bert --config configs/train_bert_cpu_sanity.yaml
```

### 5. Compare all saved runs

```bash
python -m scripts.compare_models
```

The comparison uses fresh `metrics.json` artifacts when available and falls back
to explicitly configured previous-run metrics from `configs/previous_model_metrics.json`.

Export README-ready plots from saved metrics:

```bash
python -m scripts.export_eval_plots
```

To log training params, metrics, and local artifacts to MLflow, set
`track_mlflow: true` and a local SQLite or remote `mlflow_tracking_uri` in the
training config. For example:

```yaml
track_mlflow: true
mlflow_tracking_uri: sqlite:///mlflow.db
```

### 6. Build the LangChain + ChromaDB RAG index

```bash
python -m scripts.build_rag_index
```

The default config builds a local Chroma vector database over the full corpus using LangChain `HuggingFaceEmbeddings`.
If you want a faster local smoke run, set `sample_size:` in `configs/build_rag_index.yaml`.

### 7. Ask the indexed reviews from CLI

```bash
python -m scripts.run_rag_demo
```

### 8. Launch API and demo

```bash
python -m scripts.run_api
```

Open:

- `http://127.0.0.1:8000/` for the browser demo
- `http://127.0.0.1:8000/docs` for Swagger UI

## Config-driven workflow

All main entrypoints read YAML configs from `configs/`.

- `configs/train_baseline.yaml`
- `configs/train_binary_baseline.yaml`
- `configs/train_bert.yaml`
- `configs/train_bert_cpu_sanity.yaml`
- `configs/train_bert_custom.yaml`
- `configs/compare_models.yaml`
- `configs/previous_model_metrics.json`
- `configs/export_eval_plots.yaml`
- `configs/build_rag_index.yaml`
- `configs/rag_demo.yaml`
- `configs/api.yaml`

You can keep defaults in config files and still override them by editing YAML before launch.

## API

### `GET /health`

Returns service status and whether the local classifier and RAG index are available.

### `POST /predict`

Request:

```json
{
  "text": "Приложение стало работать заметно лучше после обновления"
}
```

### `POST /ask`

Request:

```json
{
  "question": "На что чаще всего жалуются после обновления?",
  "top_k": 5,
  "sentiment_focus": "negative",
  "generation_mode": "auto"
}
```

Response shape:

```json
{
  "question": "На что чаще всего жалуются после обновления?",
  "answer": "Found 5 relevant reviews for: ...",
  "sentiment_focus": "negative",
  "generation_mode": "extractive",
  "llm_used": false,
  "keyphrases": ["последнее обновление", "медленно работает"],
  "label_distribution": {
    "negative": 5
  },
  "contexts": [
    {
      "text": "После обновления приложение стало медленным.",
      "label_id": 0,
      "label": "negative",
      "split": "train",
      "score": 0.91
    }
  ]
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
It now includes separate forms for both `/predict` and `/ask`, so the UI and backend stay aligned.
RAG is exposed through `/ask`, Swagger UI, and the CLI demo script.

## Optional LLM generation for RAG

The retrieval layer uses LangChain + ChromaDB and works out of the box without any extra credentials.
If you want the final answer to be generated by an LLM instead of the built-in extractive summary,
set one of these modes in the request or in `configs/api.yaml`:

- `auto` - use an LLM if configured, otherwise fall back to extractive mode
- `llm` - prefer the LLM, but still fall back safely if it is unavailable
- `extractive` - use only local retrieval and summary logic

Supported environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `RAG_GENERATION_MODE`

LLM calls use LangChain `ChatOpenAI`.
This also works with OpenAI-compatible local servers such as Ollama or LM Studio if you point `OPENAI_BASE_URL` to them.

## Testing and linting

```bash
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pytest
```

## Docker

Build and run:

```bash
docker build -t nlp-pet-project .
docker run --rm -p 8000:8000 -e MODEL_DIR=artifacts/baseline nlp-pet-project
```

The container expects model files to be available inside the image or mounted at runtime.

## Notes on artifacts

- Dataset cache under `data/cache/` is ignored by git.
- Model checkpoints and evaluation artifacts under `artifacts/` are ignored by git.
- The RAG index under `artifacts/rag/` is also ignored by git and should be rebuilt locally.
- The repository tracks code, configs, and reproducible instructions instead of large binaries.

## Next ideas

- add a LangChain retrieval chain with streaming answer generation
