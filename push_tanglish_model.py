from huggingface_hub import HfApi

REPO_ID = "raaagul/tanglish-offensive-detector-onnx"

api = HfApi()
api.create_repo(REPO_ID, exist_ok=True)
api.upload_folder(
    folder_path="./onnx-model-tanglish",
    repo_id=REPO_ID,
)
print(f"Tanglish model uploaded to: https://huggingface.co/{REPO_ID}")