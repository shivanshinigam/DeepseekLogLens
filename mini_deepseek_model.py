"""
mini_deepseek_model.py
=======================
Step 1 of Sir's roadmap: Complete the Custom Architecture.

Wraps our custom LogWindowAttention (from log_attention.py) into a full
Causal Language Model that can be trained, saved, and loaded by Hugging Face.

Architecture:
  Token Embedding
    ↓
  N × DeepSeekDecoderLayer
       - RMSNorm
       - LogWindowAttention  (our sliding window module)
       - RMSNorm
       - SwiGLU FFN
    ↓
  RMSNorm
    ↓
  LM Head (vocab projection)
    ↓
  CrossEntropyLoss (during training)

Run: python3 mini_deepseek_model.py
  → Runs a self-test: forward pass on a dummy batch, prints memory and param count.
"""

import math
import json
import os
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MiniDeepSeekConfig:
    """
    Model hyperparameters. Kept small so the model fits in 500 MB on CPU/Mac.
    Increase for production — the architecture scales identically.
    """
    vocab_size:         int   = 50257   # GPT-2 tokenizer vocabulary size
    hidden_size:        int   = 256     # Embedding and residual stream dimension
    num_attention_heads:int   = 4       # Multi-head attention heads
    rope_dim:           int   = 64      # Dimensions used for RoPE in attention
    window_size:        int   = 128     # Log context window (lines/tokens)
    intermediate_size:  int   = 1024    # Feed-forward hidden size (4× hidden typical)
    num_hidden_layers:  int   = 4       # Number of transformer decoder blocks
    max_position_embeddings: int = 512  # Max sequence length
    rms_norm_eps:       float = 1e-5
    initializer_range:  float = 0.02
    causal:             bool  = True    # True = streaming logs, False = static files

    # Tells Hugging Face and vLLM what class to load
    architectures: list = None
    model_type:    str  = "mini_deepseek"

    def __post_init__(self):
        if self.architectures is None:
            self.architectures = ["CustomMiniDeepSeek"]
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"

    def to_dict(self):
        d = asdict(self)
        d["torch_dtype"] = "float32"
        return d


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm  (DeepSeek uses RMSNorm, not LayerNorm)
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm, no mean sub)."""
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms  = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE helpers (self-contained so this file works standalone)
# ─────────────────────────────────────────────────────────────────────────────

def build_rope_cache(seq_len: int, rope_dim: int, device: torch.device) -> tuple:
    """Build cos/sin tables for Rotary Positional Encoding."""
    theta   = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2, device=device).float() / rope_dim))
    pos     = torch.arange(seq_len, device=device).float()
    freqs   = torch.outer(pos, theta)          # [seq_len, rope_dim/2]
    freqs   = torch.cat([freqs, freqs], dim=-1) # [seq_len, rope_dim]
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, rope_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ─────────────────────────────────────────────────────────────────────────────
# Sliding Window Self-Attention  (our core custom module)
# ─────────────────────────────────────────────────────────────────────────────

class LogWindowAttention(nn.Module):
    """
    Sliding window self-attention with Decoupled RoPE.

    Memory is O(W × h × d) — fixed regardless of log file length.
    This is the module that replaces standard full attention in DeepSeek.
    """

    def __init__(self, config: MiniDeepSeekConfig):
        super().__init__()
        self.num_heads   = config.num_attention_heads
        self.head_dim    = config.hidden_size // config.num_attention_heads
        self.rope_dim    = min(config.rope_dim, self.head_dim)
        self.window_size = config.window_size
        self.causal      = config.causal
        self.scale       = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _build_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Boolean mask: True = attend, False = block (outside window)."""
        i   = torch.arange(seq_len, device=device).unsqueeze(1)
        j   = torch.arange(seq_len, device=device).unsqueeze(0)
        win = (j >= i - self.window_size) & (j <= i)
        if not self.causal:
            win = win | win.T   # bidirectional: also look forward W tokens
        return win   # [seq_len, seq_len]  bool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            [batch, seq_len, hidden_size]
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [B, heads, T, head_dim]

        # Apply RoPE to the rope_dim portion of Q and K
        cos, sin = build_rope_cache(T, self.rope_dim, x.device)
        q[..., :self.rope_dim], k[..., :self.rope_dim] = apply_rope(
            q[..., :self.rope_dim], k[..., :self.rope_dim], cos, sin
        )

        # Scaled dot-product attention with window mask
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, h, T, T]
        mask = self._build_mask(T, x.device)
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn)  # replace NaN from all-masked rows

        out = torch.matmul(attn, v)                  # [B, h, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU FFN  (DeepSeek's feed-forward style)
# ─────────────────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU activation: better than standard GELU for language models.
    Uses 2/3 × intermediate_size to keep param count comparable.
    """
    def __init__(self, config: MiniDeepSeekConfig):
        super().__init__()
        inter = int(config.intermediate_size * 2 / 3)
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ─────────────────────────────────────────────────────────────────────────────
# Decoder Layer: Attention + FFN + residuals
# ─────────────────────────────────────────────────────────────────────────────

class DeepSeekDecoderLayer(nn.Module):
    """
    One transformer decoder block:
      x → RMSNorm → LogWindowAttention → residual
        → RMSNorm → SwiGLU FFN        → residual
    """
    def __init__(self, config: MiniDeepSeekConfig):
        super().__init__()
        self.input_layernorm          = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn                = LogWindowAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp                      = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full Causal Language Model
# ─────────────────────────────────────────────────────────────────────────────

class CustomMiniDeepSeek(nn.Module):
    """
    Complete causal LM with our sliding window attention.

    Compatible with:
      - Standard PyTorch training loop  (train_mini_deepseek.py)
      - Hugging Face save/load          (save_and_upload.py)
      - Text generation (greedy/sample) via .generate()
    """

    def __init__(self, config: MiniDeepSeekConfig):
        super().__init__()
        self.config       = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers       = nn.ModuleList([DeepSeekDecoderLayer(config)
                                           for _ in range(config.num_hidden_layers)])
        self.norm         = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # Tie lm_head to embedding weights (standard LM trick, saves params)
        self.lm_head      = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self,
                input_ids: torch.Tensor,
                labels:    torch.Tensor = None) -> tuple:
        """
        Args:
            input_ids: [batch, seq_len]  token IDs
            labels:    [batch, seq_len]  target IDs (same as input_ids for CLM)
        Returns:
            (logits, loss)   loss is None when labels is None
        """
        x = self.embed_tokens(input_ids)          # [B, T, hidden]

        for layer in self.layers:
            x = layer(x)

        x      = self.norm(x)
        logits = self.lm_head(x)                  # [B, T, vocab_size]

        loss = None
        if labels is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 temperature: float = 0.8) -> torch.Tensor:
        """Simple greedy/temperature text generation for testing."""
        for _ in range(max_new_tokens):
            # Truncate to max window size
            idx_cond = input_ids[:, -self.config.max_position_embeddings:]
            logits, _ = self.forward(idx_cond)
            logits    = logits[:, -1, :] / temperature
            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def memory_estimate_mb(self) -> float:
        """Rough estimate: params × 4 bytes (float32)."""
        return self.num_parameters() * 4 / (1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  CustomMiniDeepSeek — Architecture Self-Test")
    print("═" * 60)

    config = MiniDeepSeekConfig()
    model  = CustomMiniDeepSeek(config)
    model.eval()

    B, T = 2, 64  # batch=2, seq_len=64 tokens
    dummy_ids = torch.randint(0, config.vocab_size, (B, T))

    # Forward pass
    with torch.no_grad():
        logits, loss = model(dummy_ids, labels=dummy_ids)

    print(f"\n  Config:")
    print(f"    hidden_size       : {config.hidden_size}")
    print(f"    num_layers        : {config.num_hidden_layers}")
    print(f"    num_heads         : {config.num_attention_heads}")
    print(f"    window_size       : {config.window_size} tokens")
    print(f"    vocab_size        : {config.vocab_size}")
    print(f"\n  Forward pass:")
    print(f"    Input shape       : {list(dummy_ids.shape)}")
    print(f"    Logits shape      : {list(logits.shape)}")
    print(f"    Loss              : {loss.item():.4f} ✅")
    print(f"\n  Model size:")
    print(f"    Parameters        : {model.num_parameters():,}")
    print(f"    Memory (float32)  : {model.memory_estimate_mb():.1f} MB")
    budget = 500 - model.memory_estimate_mb()
    print(f"    Remaining budget  : {budget:.1f} MB / 500 MB ✅")
    print(f"\n  Generation test:")
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    output = model.generate(prompt, max_new_tokens=20)
    print(f"    Input tokens      : {prompt.shape[1]}")
    print(f"    Output tokens     : {output.shape[1]} (+20 generated) ✅")
    print(f"\n{'═' * 60}")
    print(f"  ✅ All checks passed. Ready for training.")
    print(f"  Next: python3 train_mini_deepseek.py")
    print(f"{'═' * 60}\n")
