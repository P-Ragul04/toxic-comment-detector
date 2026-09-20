from huggingface_hub import HfApi

REPO_ID = "raaagul/toxic-comment-detector-onnx"

api = HfApi()
api.create_repo(REPO_ID, exist_ok=True)
api.upload_folder(
    folder_path="./onnx-model",
    repo_id=REPO_ID,
)
print(f"ONNX model uploaded to: https://huggingface.co/{REPO_ID}")