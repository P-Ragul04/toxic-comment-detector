# Toxic Comment Detector

A multilingual content moderation system with two fine-tuned DistilBERT models — one for English (6-category multi-label toxicity), one for Tanglish/romanized Tamil-English (binary offensive-language detection) — served via a FastAPI backend and a Streamlit frontend, fully containerized and deployed on free-tier infrastructure.

**Live demo:** [toxic-comment-detector-z8q2.onrender.com/docs](https://toxic-comment-detector-z8q2.onrender.com/docs)

\---

## What it does

Given a piece of text, the system predicts:

* **English mode:** probability across 6 categories — `toxic`, `severe\_toxic`, `obscene`, `threat`, `insult`, `identity\_hate` — each flagged independently against a per-label tuned decision threshold.
* **Tanglish mode:** binary classification — `offensive` / `not offensive` — for romanized Tamil-English text (e.g. *"nee oru periya loosu paiyan da"*), since that's how Tamil speakers actually write online, not in native script.

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌───────────────────────┐
│  Streamlit UI    │─────▶│   FastAPI          │─────▶│  ONNX Runtime          │
│  (local/browser)  │      │  /predict           │      │  (quantized DistilBERT) │
│                   │      │  /predict-tanglish  │      │  x2 models              │
└─────────────────┘      └──────────────────┘      └───────────────────────┘
                                    │
                                    ▼
                          Hugging Face Hub
                    (model weights, pulled at container startup)
```

The API is Dockerized and deployed on **Render** (free tier). Model weights aren't baked into the Docker image — they're pulled from Hugging Face Hub at container startup, keeping the image small and letting models be updated without rebuilding.

## Models

||English toxicity|Tanglish offensive|
|-|-|-|
|Base model|`distilbert-base-uncased`|`distilbert-base-uncased`|
|Task|Multi-label (6 classes)|Binary|
|Training data|[Jigsaw Toxic Comment Classification](https://huggingface.co/datasets/jigsaw_toxicity_pred) (\~160k comments)|[offenseval\_dravidian](https://huggingface.co/datasets/community-datasets/offenseval_dravidian) (Tamil subset, filtered to Latin-script rows)|
|Macro F1|0.61 (default threshold) → 0.605 (tuned)|0.79|
|Deployment format|ONNX, int8 quantized|ONNX, int8 quantized|
|Model size (quantized)|64 MB|64 MB|

Both models are hosted on Hugging Face Hub:

* [`raaagul/toxic-comment-detector-onnx`](https://huggingface.co/raaagul/toxic-comment-detector-onnx)
* [`raaagul/tanglish-offensive-detector-onnx`](https://huggingface.co/raaagul/tanglish-offensive-detector-onnx)

## Key engineering decisions

**Why DistilBERT for both, instead of a multilingual model for Tanglish?**
Tanglish is written in Latin script, so DistilBERT's existing English-vocabulary tokenizer can process it as subword tokens without needing an architecture change. A multilingual model (e.g. XLM-RoBERTa) would only be necessary for native Tamil script, which is rarely how offensive comments are actually typed online.

**Why per-label threshold tuning instead of a flat 0.5 cutoff?**
The English dataset is heavily imbalanced (e.g. `threat` has \~200 positive examples vs. `toxic`'s 6,000+). A single global threshold underserves rare labels. Thresholds were tuned on a held-out validation split (never the same data used for final reported metrics, to avoid overfitting the threshold itself) — this improved micro F1 meaningfully (0.675 → 0.701) while honestly reporting that tuning didn't help — and in one case (`threat`) slightly hurt — the rarest labels, since there wasn't enough data to estimate a stable threshold for them.

**Why ONNX Runtime instead of PyTorch in production?**
The original PyTorch + Transformers deployment exceeded Render's free-tier 512MB memory limit before serving a single request — full PyTorch's import footprint alone is substantial. Converting to ONNX (an inference-only runtime with no training machinery) plus int8 dynamic quantization cut the English model from 268MB → 64MB and dropped the total memory footprint enough to fit comfortably, even with two models loaded simultaneously.

## Tech stack

* **Modeling:** PyTorch, Hugging Face Transformers, scikit-learn
* **Serving:** FastAPI, ONNX Runtime, Pydantic
* **Frontend:** Streamlit
* **Infra:** Docker, GitHub Actions (CI: tests + Docker build on every push), Render (API hosting), Hugging Face Hub (model hosting)
* **Testing:** pytest

## Project structure

```
├── training/
│   ├── train.py                    # English model fine-tuning
│   ├── train\_tanglish.py           # Tanglish model fine-tuning
│   ├── evaluate.py                 # Detailed per-label metrics
│   ├── tune\_thresholds.py          # Non-cheating threshold tuning (English)
│   ├── tune\_thresholds\_tanglish.py # Threshold tuning (Tanglish)
│   ├── export\_onnx.py              # ONNX conversion + quantization (English)
│   └── export\_onnx\_tanglish.py     # ONNX conversion + quantization (Tanglish)
├── api/
│   ├── main.py                     # FastAPI app, both endpoints
│   ├── model.py                    # English model loading/inference
│   ├── tanglish\_model.py           # Tanglish model loading/inference
│   ├── schemas.py                  # Request/response validation
│   ├── Dockerfile
│   └── tests/test\_api.py
├── frontend/
│   └── app.py                      # Streamlit UI
└── .github/workflows/ci.yml        # Tests + Docker build on every push
```

## Running locally

```bash
# API
cd api
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (separate terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

This is a portfolio project, not a production moderation tool. Both models can make mistakes — the English model especially on rare categories (`threat`, `severe\_toxic`) where training data was limited, and the Tanglish model on informal or ambiguous phrasing where even human annotators in the original dataset often disagreed. The free-tier API also spins down after inactivity, so the first request after idle time can take up to \~50 seconds to respond.

## 

