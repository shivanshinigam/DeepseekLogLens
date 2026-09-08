"""
save_and_upload.py
===================
Phase 1 of the RunPod roadmap: Upload trained weights to Hugging Face Hub.

After running train_mini_deepseek.py, this script:
  1. Verifies the saved model loads correctly
  2. Creates a private HuggingFace repository
  3. Uploads all model files (weights, config, tokenizer)

Usage:
    python3 save_and_upload.py --repo-id your-username/loglens-mini-deepseek

Prerequisites:
    pip3 install huggingface_hub
    huggingface-cli login   (paste your Write token from https://hf.co/settings/tokens)
"""

import os
import json
import argparse

import torch
from transformers import AutoTokenizer
from mini_deepseek_model import CustomMiniDeepSeek, MiniDeepSeekConfig


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Verify saved model
# ─────────────────────────────────────────────────────────────────────────────

def verify_model(model_dir: str) -> bool:
    """Load the saved model and run a forward pass to confirm it's not corrupted."""
    print(f"\n🔍 Verifying saved model at {model_dir}/…")

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"  ❌ config.json not found. Run train_mini_deepseek.py first.")
        return False

    with open(config_path) as f:
        cfg_dict = json.load(f)

    config = MiniDeepSeekConfig(
        vocab_size          = cfg_dict["vocab_size"],
        hidden_size         = cfg_dict["hidden_size"],
        num_attention_heads = cfg_dict["num_attention_heads"],
        num_hidden_layers   = cfg_dict["num_hidden_layers"],
        intermediate_size   = cfg_dict["intermediate_size"],
    )
    model = CustomMiniDeepSeek(config)

    # Load weights
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    bin_path         = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        print(f"  ✅ Loaded weights from model.safetensors")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
        print(f"  ✅ Loaded weights from pytorch_model.bin")
    else:
        print(f"  ❌ No weight file found in {model_dir}/")
        return False

    model.load_state_dict(state_dict, strict=False)
    # strict=False is correct here: lm_head.weight is tied to embed_tokens.weight
    # and safetensors saves it only once, so lm_head.weight appears "missing" to strict=True
    model.eval()

    # Quick forward pass
    dummy = torch.randint(0, config.vocab_size, (1, 32))
    with torch.no_grad():
        logits, _ = model(dummy)

    print(f"  ✅ Forward pass OK — logits shape: {list(logits.shape)}")
    print(f"  ✅ Parameters: {model.num_parameters():,} | Memory: {model.memory_estimate_mb():.1f} MB")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Upload to Hugging Face
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_hf(model_dir: str, repo_id: str, private: bool = True):
    """
    Create a HuggingFace repo and upload all files from model_dir.

    Args:
        model_dir : local folder with weights, config.json, tokenizer files
        repo_id   : "your-username/your-model-name"
        private   : True = private repo (recommended for proprietary models)
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  ❌ huggingface_hub not installed.")
        print("     Run: pip3 install huggingface_hub && huggingface-cli login")
        return

    api = HfApi()

    print(f"\n📤 Uploading to Hugging Face: {repo_id}")
    print(f"   Visibility: {'private 🔒' if private else 'public 🌐'}")

    # Create repo (won't fail if it already exists)
    api.create_repo(
        repo_id   = repo_id,
        repo_type = "model",
        private   = private,
        exist_ok  = True
    )
    print(f"  ✅ Repository created/verified: https://huggingface.co/{repo_id}")

    # Upload the entire local folder
    print(f"  ⏳ Uploading files from {model_dir}/ …")
    api.upload_folder(
        folder_path = model_dir,
        repo_id     = repo_id,
        repo_type   = "model",
        commit_message = "Upload CustomMiniDeepSeek weights, config, and tokenizer"
    )

    print(f"\n  ✅ Upload complete!")
    print(f"  🔗 Model URL: https://huggingface.co/{repo_id}")
    print(f"\n  Next step → Set up RunPod vLLM template (see docs/runpod_deployment.md)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and upload trained model to HuggingFace")
    parser.add_argument("--model-dir", type=str, default="./my_custom_mini_deepseek",
                        help="Local directory containing trained model files")
    parser.add_argument("--repo-id",   type=str, default=None,
                        help="HuggingFace repo ID e.g. shivanshinigam/loglens-mini-deepseek")
    parser.add_argument("--public",    action="store_true",
                        help="Make the HuggingFace repo public (default: private)")
    args = parser.parse_args()

    # Always verify first
    ok = verify_model(args.model_dir)

    if not ok:
        print("\n❌ Verification failed. Fix errors above before uploading.")
        exit(1)

    if args.repo_id:
        upload_to_hf(args.model_dir, args.repo_id, private=not args.public)
    else:
        print(f"\n  ✅ Verification passed.")
        print(f"  To upload, run:")
        print(f"     python3 save_and_upload.py --repo-id your-username/loglens-mini-deepseek")
        print(f"  Make sure to login first: huggingface-cli login")
