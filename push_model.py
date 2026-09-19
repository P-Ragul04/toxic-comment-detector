from huggingface_hub import HfApi

REPO_ID = "raaagul/toxic-comment-detector"

api = HfApi()
api.create_repo(REPO_ID, exist_ok=True)
api.upload_folder(
    folder_path="./toxic-detector-model",
    repo_id=REPO_ID,
)
print(f"Model uploaded to: https://huggingface.co/{REPO_ID}")