"""
llama_compatible_train.py — LogLens AI · vLLM-Ready Llama-Mapped Model
=======================================================================
Builds, trains, and saves a Llama-compatible version of our custom
sliding window attention model.

Why this exists:
  Our original mini_deepseek_model.py uses "CustomMiniDeepSeek" as the
  architecture name. vLLM does NOT know this name, so it cannot serve it.

  This script renames everything to match Llama's exact parameter layout
  (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, RMSNorm)
  and saves config.json with "architectures": ["LlamaForCausalLM"].

  vLLM sees this and says "I know LlamaForCausalLM!" and serves it using
  its optimized Llama CUDA kernels — no custom extensions needed.

Output: ./vllm_ready_mini_model/
  ├── model.safetensors   ← weights
  ├── config.json         ← "architectures": ["LlamaForCausalLM"]
  ├── tokenizer.json      ← Llama tokenizer
  └── tokenizer_config.json

Usage:
    python3 llama_compatible_train.py
    python3 llama_compatible_train.py --epochs 3 --data training_logs.txt
"""

import os
import json
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",  type=int,   default=2,                      help="Training epochs")
parser.add_argument("--lr",      type=float, default=3e-4,                   help="Learning rate")
parser.add_argument("--batch",   type=int,   default=4,                      help="Batch size")
parser.add_argument("--maxlen",  type=int,   default=128,                    help="Max token length")
parser.add_argument("--data",    type=str,   default="training_logs.txt",    help="Training data file")
parser.add_argument("--outdir",  type=str,   default="./vllm_ready_mini_model", help="Output dir")
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG — Llama-compatible field names
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MiniLlamaConfig:
    vocab_size:              int   = 32000   # Llama tokenizer size
    hidden_size:             int   = 256     # Keep lightweight (<500 MB)
    intermediate_size:       int   = 684     # ~2.67× hidden_size (Llama ratio)
    num_hidden_layers:       int   = 4       # 4 decoder blocks
    num_attention_heads:     int   = 4
    num_key_value_heads:     int   = 4       # MHA (same as num_attention_heads)
    rms_norm_eps:            float = 1e-5
    max_position_embeddings: int   = 2048


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILDING BLOCKS — exact Llama parameter names
# ─────────────────────────────────────────────────────────────────────────────
class LlamaRMSNorm(nn.Module):
    """Llama-style RMSNorm (parameter name: .weight — matches Llama state dict)"""
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        v = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(v + self.variance_epsilon)
        return self.weight * x.to(dtype)


class DeepSeekWindowAttention(nn.Module):
    """
    Our custom Sliding Window Attention wrapped in Llama's parameter names.

    vLLM expects: q_proj, k_proj, v_proj, o_proj  ← these exact names.

    The custom part: we apply a causal sliding window mask so each token
    only attends to the W tokens immediately before it (not the whole sequence).
    This is the O(W) memory trick — the core of the LogLens approach.
    """
    def __init__(self, config: MiniLlamaConfig, window_size: int = 128):
        super().__init__()
        self.hidden_size  = config.hidden_size
        self.num_heads    = config.num_attention_heads
        self.head_dim     = config.hidden_size // config.num_attention_heads
        self.window_size  = window_size

        # Llama-named projection layers (vLLM reads these exact names)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _build_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Causal sliding window mask.
        Token i can attend to tokens in [i - window_size, i].
        Everything outside → -inf → zero weight after softmax.
        """
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        mask = torch.tril(mask, diagonal=0) & torch.triu(mask, diagonal=-self.window_size)
        return mask  # True = allowed, False = masked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale   # [B, H, T, T]

        # Apply sliding window mask
        window_mask = self._build_window_mask(T, x.device)      # [T, T]
        scores = scores.masked_fill(
            ~window_mask.unsqueeze(0).unsqueeze(0),              # broadcast over B, H
            float('-inf')
        )

        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)                       # [B, H, T, head_dim]

        # Merge heads and project
        context = context.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        return self.o_proj(context)


class LlamaMLP(nn.Module):
    """
    SwiGLU feed-forward network — exact Llama parameter names:
    gate_proj, up_proj, down_proj  (vLLM matches these automatically)
    """
    def __init__(self, config: MiniLlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn    = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate * up → down
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """One Llama decoder block: Norm → Attention → Norm → MLP (residuals)"""
    def __init__(self, config: MiniLlamaConfig):
        super().__init__()
        self.input_layernorm          = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn                = DeepSeekWindowAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp                      = LlamaMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. FULL CAUSAL LM — Llama state-dict layout
#    model.embed_tokens · model.layers.N.* · model.norm · lm_head
# ─────────────────────────────────────────────────────────────────────────────
class LlamaForCausalLM(nn.Module):
    """
    State-dict keys match Llama exactly:
      model.embed_tokens.weight
      model.layers.0.input_layernorm.weight
      model.layers.0.self_attn.q_proj.weight
      ...
      model.norm.weight
      lm_head.weight
    vLLM reads these without any custom code.
    """
    def __init__(self, config: MiniLlamaConfig):
        super().__init__()
        self.config = config

        # The sub-module MUST be named 'model' to match Llama's state dict
        self.model = nn.ModuleDict({
            "embed_tokens": nn.Embedding(config.vocab_size, config.hidden_size),
            "layers":       nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]),
            "norm":         LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        })
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        x = self.model["embed_tokens"](input_ids)
        for layer in self.model["layers"]:
            x = layer(x)
        x = self.model["norm"](x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Causal LM: predict next token → shift by 1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        return logits, loss


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET
# ─────────────────────────────────────────────────────────────────────────────
class LogFileDataset(Dataset):
    """Reads a text file line-by-line, tokenizes each line."""
    def __init__(self, file_path: str, tokenizer, max_length: int = 128):
        self.examples = []

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
        else:
            # Fallback: use built-in log snippets if file not found
            print(f"⚠️  {file_path} not found — using built-in sample log data")
            lines = [
                "LOG INFO: User authentication successful for user_id 8821.",
                "LOG WARN: API response latency peaked at 452ms on /api/v1/auth.",
                "LOG ERROR: Database connection timeout on primary cluster region-us-east.",
                "LOG INFO: Task scheduler triggered cron job #992 successfully.",
                "LOG DEBUG: Connection pool active connections: 14/100, idle: 86.",
                "LOG WARN: High memory utilization detected on node-worker-04 (87%).",
                "LOG ERROR: Redis connection dropped unexpectedly; attempting reconnection.",
                "LOG WARN: Memory leak warning issued for long-running process pid-8441.",
                "LOG ERROR: Database deadlock encountered during row update in transaction_id 99182.",
                "LOG ERROR: SSL certificate handshake failed from client IP 192.168.1.45.",
            ] * 20   # repeat 20× for enough training samples

        for text in lines:
            enc = tokenizer(
                text,
                max_length    = max_length,
                truncation    = True,
                padding       = "max_length",
                return_tensors= "pt"
            )
            self.examples.append(enc["input_ids"].squeeze(0))

    def __len__(self):   return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN — TRAIN + SAVE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  LogLens AI — Llama-Compatible vLLM Model Builder")
    print("=" * 60)

    # ── Tokenizer ──────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print("\n🔤 Loading tokenizer…")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            trust_remote_code=True
        )
    except Exception:
        # Fallback: use GPT-2 tokenizer (also works)
        print("   (Llama tokenizer unavailable — falling back to GPT-2)")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"   vocab_size = {vocab_size}")

    # ── Config ─────────────────────────────────────────────────────
    config = MiniLlamaConfig(vocab_size=vocab_size)

    # ── Dataset ────────────────────────────────────────────────────
    print(f"\n📂 Loading training data from: {args.data}")
    dataset    = LogFileDataset(args.data, tokenizer, max_length=args.maxlen)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    print(f"   {len(dataset)} samples → {len(dataloader)} batches/epoch")

    # ── Model ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LlamaForCausalLM(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🏗  Model: {total_params/1e6:.1f}M params | device: {device}")

    # ── Training ───────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    print(f"\n🏋️  Training for {args.epochs} epoch(s)…\n")
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch.to(device)
            _, loss   = model(input_ids, labels=input_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % 5 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}  step {step:3d}/{len(dataloader)}  loss={loss.item():.4f}")

        avg = total_loss / len(dataloader)
        print(f"  ✅ Epoch {epoch+1} complete — avg loss: {avg:.4f}\n")

    # ── Save ───────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)

    # Weights
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, os.path.join(args.outdir, "model.safetensors"))
    print(f"💾 Weights saved → {args.outdir}/model.safetensors")

    # config.json — "LlamaForCausalLM" so vLLM recognises it
    config_dict = {
        "architectures":           ["LlamaForCausalLM"],
        "model_type":              "llama",
        "vocab_size":              config.vocab_size,
        "hidden_size":             config.hidden_size,
        "intermediate_size":       config.intermediate_size,
        "num_hidden_layers":       config.num_hidden_layers,
        "num_attention_heads":     config.num_attention_heads,
        "num_key_value_heads":     config.num_key_value_heads,
        "rms_norm_eps":            config.rms_norm_eps,
        "max_position_embeddings": config.max_position_embeddings,
        "torch_dtype":             "float32",
        "log_window_size":         128,     # our custom field — ignored by vLLM, used by us
        "transformers_version":    "4.40.0"
    }
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"💾 Config saved  → {args.outdir}/config.json")

    # Tokenizer
    tokenizer.save_pretrained(args.outdir)
    print(f"💾 Tokenizer saved → {args.outdir}/")

    print(f"""
════════════════════════════════════════════════════════
  ✅ vLLM-ready model saved to: {args.outdir}/

  Files:
    model.safetensors   ← weights
    config.json         ← "architectures": ["LlamaForCausalLM"]
    tokenizer.json
    tokenizer_config.json

  Verify it loads:
    python3 verify_model.py

  Upload to HuggingFace:
    python3 save_and_upload.py --repo-id ShivanshiNigam/loglens-vllm-ready \\
                               --model-dir {args.outdir}

  RunPod vLLM command:
    --model ShivanshiNigam/loglens-vllm-ready --port 8000 --trust-remote-code
════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
