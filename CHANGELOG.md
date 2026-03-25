# Changelog

## Stage 1 - Data and baselines

- cached the RuReviews dataset locally and added text cleaning utilities
- built a TF-IDF + LogisticRegression baseline to anchor later experiments
- added evaluation artifacts such as metrics, confusion matrix, and error tables

## Stage 2 - Transformer training

- trained ruBERT with both Hugging Face Trainer and a custom PyTorch loop
- tuned the Trainer setup to improve macro F1 over the initial run
- added checkpoint-based inference for local predictions

## Stage 3 - Product surface

- exposed sentiment inference through FastAPI
- added a browser demo for manual inspection and smoke testing
- added tests, Docker, CI, and config-driven entrypoints

## Stage 4 - Review intelligence

- indexed the review corpus with sentence-transformer embeddings
- added retrieval QA over the corpus through CLI and FastAPI
- prepared the project for optional LLM-backed answer synthesis on top of retrieved reviews
