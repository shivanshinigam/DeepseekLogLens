"""
upload_to_hf.py — LogLens AI
=============================
Uploads the vLLM-ready mini model to Hugging Face Hub.

Usage:
    python3 upload_to_hf.py
"""

from huggingface_hub import HfApi
import sys

def main():
    repo_id = "ShivanshiNigam/loglens-vllm-ready"
    folder_path = "./vllm_ready_mini_model"
    
    print(f"📤 Uploading {folder_path} to Hugging Face: {repo_id}")

    try:
        api = HfApi()

        # Create the repository on Hugging Face
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        print("✅ Repository verified.")

        # Upload the entire local folder containing your weights
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload Llama-mapped vLLM-ready model weights and config"
        )
        print(f"✅ Successfully uploaded to {repo_id}")
        print(f"🔗 https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("Did you run `huggingface-cli login`?")
        sys.exit(1)

if __name__ == "__main__":
    main()
