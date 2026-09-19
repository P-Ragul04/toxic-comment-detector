# Week 1 Setup: Training the model locally

## 1. Environment setup

```bash
# from the training/ folder
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

# Install PyTorch with CUDA support (important - the default pip install
# may give you a CPU-only build). Check your CUDA version first:
nvidia-smi
```

Then grab the matching PyTorch command from https://pytorch.org/get-started/locally/
For most recent NVIDIA drivers, this works:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is detected:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Should print `True RTX 3050 Ti Laptop GPU` (or similar).

## 2. Get the dataset

The Jigsaw dataset can't be auto-downloaded (Kaggle competition licensing),
so it needs a manual step:

1. Go to https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge
   (this is a clean re-upload of the original competition data — free Kaggle account required)
2. Download the dataset
3. Extract it into `training/jigsaw_data/` so you have:
   ```
   training/jigsaw_data/train.csv
   training/jigsaw_data/test.csv
   training/jigsaw_data/test_labels.csv
   ```

If `load_dataset("jigsaw_toxicity_pred", data_dir="./jigsaw_data")` still
complains about format, let me know what error you get and I'll adjust
the loading code — HF's dataset script for this one is a bit finicky
about exact file naming.

## 3. Run training

```bash
cd training
python train.py
```

On an RTX 3050 Ti (6GB), 3 epochs over ~160k comments at max_length=128
should take roughly 1.5-3 hours depending on thermals/throttling. If you
hit a CUDA out-of-memory error, lower `per_device_train_batch_size` in
train.py from 16 to 8.

Progress is logged to `./logs` — you can watch it live with:
```bash
tensorboard --logdir ./logs
```

## 4. Evaluate

```bash
python evaluate.py
```

This gives you per-label precision/recall/F1, which is the good stuff
to screenshot for your README later — way more informative than a bare
accuracy score, especially since this dataset is heavily imbalanced.

## 5. Push to Hugging Face Hub (do this once training looks good)

```bash
pip install huggingface_hub
huggingface-cli login   # free account, paste your token

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('your-username/toxic-comment-detector', exist_ok=True)
api.upload_folder(folder_path='./toxic-detector-model', repo_id='your-username/toxic-comment-detector')
"
```

This is what the FastAPI service will pull the model from in Week 2,
instead of you having to copy model files around manually.
