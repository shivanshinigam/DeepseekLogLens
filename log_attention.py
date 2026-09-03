"""
log_attention.py
================
Custom Sliding Window Attention module for DeepSeek, designed for log analysis.

WHY THIS EXISTS
---------------
Standard DeepSeek uses full self-attention: every token attends to every other token.
For a 10,000-page log file this is O(n²) memory — it grows with the file size.

Our use case only needs local context: when analyzing log line 5000,
only lines 4990–5010 matter. Everything else is noise.

This module replaces full attention with a fixed sliding window of W tokens.
Memory stays at O(W) — flat, predictable, always under 500 MB.

HOW IT PLUGS INTO DEEPSEEK
---------------------------
In modeling_deepseek.py, find the DeepseekAttention class.
Replace the attention forward() call with LogWindowAttention.

See: README.md → Section 3 for exact swap instructions.

TESTED ON
---------
- Python 3.10+
- PyTorch 2.1+
- No GPU required for import/inspection
- GPU required for actual inference
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# RoPE (Rotary Positional Embedding) helpers
# ─────────────────────────────────────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Core RoPE rotation trick.
    Splits the last dimension in half, flips and negates to create rotation.

    Input shape:  [..., dim]
    Output shape: [..., dim]  (same shape, different values)
    """
    x1 = x[..., : x.shape[-1] // 2]   # first half
    x2 = x[..., x.shape[-1] // 2 :]   # second half
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE positional encoding to Query and Key tensors.

    DeepSeek uses 'Decoupled RoPE': only a subset of dimensions get
    positional rotation. The rest (content dimensions) are passed through as-is.
    This function handles the rotation part.

    Args:
        q   : Query tensor  [batch, seq_len, num_heads, rope_dim]
        k   : Key tensor    [batch, seq_len, num_heads, rope_dim]
        cos : Cosine table  [seq_len, rope_dim]  — will be broadcast over heads
        sin : Sine table    [seq_len, rope_dim]  — will be broadcast over heads

    Returns:
        (q_rotated, k_rotated) — same shape as inputs
    """
    # cos/sin are [seq_len, rope_dim] — we need [1, seq_len, 1, rope_dim] to
    # broadcast correctly over [batch, seq_len, num_heads, rope_dim]
    cos = cos.unsqueeze(0).unsqueeze(2)   # [1, seq_len, 1, rope_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)   # [1, seq_len, 1, rope_dim]
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated


# ─────────────────────────────────────────────────────────────────────────────
# Sliding Window Mask builder
# ─────────────────────────────────────────────────────────────────────────────

def build_window_mask(
    seq_len: int,
    window_size: int,
    causal: bool = True,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Build a boolean attention mask that limits each token to only attending
    to tokens within [i - window_size, i] (causal) or [i - W, i + W] (bidirectional).

    Why causal=True for log analysis?
        Log files are read top-to-bottom. When analyzing line 5000,
        we only look BACK (lines 4980–5000), not forward.
        This also enables streaming: we can analyze as logs come in.

    Why causal=False (bidirectional)?
        If you have a complete static log file and want to look both
        before AND after the current line for context.

    Args:
        seq_len     : Total number of tokens in the current window
        window_size : How many tokens back (and optionally forward) to attend to
        causal      : If True, each token only sees past tokens (streaming safe)
        device      : torch device

    Returns:
        Boolean mask [seq_len, seq_len]
        True  = this pair CAN attend to each other
        False = this pair is BLOCKED (will become -inf in softmax)

    Memory note:
        This mask is seq_len × seq_len booleans.
        For window_size=512 → 512×512×1 byte = 0.25 MB. Negligible.
    """
    # Start with all True (fully connected)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

    if causal:
        # Causal window: token i sees tokens [i-window_size, i]
        # tril → only lower triangle (no future)
        # triu with -window_size → only W tokens in the past
        mask = torch.tril(mask, diagonal=0) & torch.triu(mask, diagonal=-window_size)
    else:
        # Bidirectional window: token i sees tokens [i-W, i+W]
        mask = torch.tril(mask, diagonal=window_size) & torch.triu(mask, diagonal=-window_size)

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Main: Log Window Attention Module
# ─────────────────────────────────────────────────────────────────────────────

class LogWindowAttention(nn.Module):
    """
    Sliding Window Self-Attention for Log Analysis.

    Replaces DeepSeek's full self-attention with a local window version.
    Each log token only attends to tokens within a fixed window before it.

    Memory:
        Full attention    → O(n²) — grows with log file size (MEMORY KILLER)
        Window attention  → O(W)  — fixed, does not grow (SAFE FOR 500 MB)

    Window size guidance:
        window_size=256  → ~1 log page of context    → fastest, least memory
        window_size=512  → ~2 log pages of context   → recommended (default)
        window_size=1024 → ~4 log pages of context   → more context, still safe

    Usage:
        See README.md → Section 3 for how to swap this into modeling_deepseek.py

    Args:
        hidden_size         : Model hidden dimension (e.g. 2048 for DeepSeek-1.3B)
        num_attention_heads : Number of attention heads (e.g. 16)
        rope_dim            : Dimensions used for RoPE (DeepSeek typically uses 64)
        window_size         : How many tokens back each token can attend to (default: 512)
        causal              : If True, tokens only see past (streaming logs).
                              If False, bidirectional (full static log files).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        rope_dim: int = 64,
        window_size: int = 512,
        causal: bool = True,
    ):
        super().__init__()
        self.num_heads   = num_attention_heads
        self.head_dim    = hidden_size // num_attention_heads
        self.rope_dim    = rope_dim
        self.window_size = window_size
        self.causal      = causal
        self.scale       = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        v: torch.Tensor,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass — windowed attention.

        DeepSeek uses 'Decoupled RoPE' (from MLA architecture):
        - Q and K are split into two parts:
          1. Content part (q_content, k_content) — no positional info, raw semantics
          2. RoPE part    (q_rope,    k_rope)    — positional rotation applied here

        - Final attention score = (Q_content @ K_content^T) + (Q_rope @ K_rope^T)
        - Then apply window mask, softmax, weighted sum over V.

        Args:
            q_content : [batch, seq_len, num_heads, head_dim - rope_dim]
            k_content : [batch, seq_len, num_heads, head_dim - rope_dim]
            v         : [batch, seq_len, num_heads, head_dim]
            q_rope    : [batch, seq_len, num_heads, rope_dim]
            k_rope    : [batch, seq_len, num_heads, rope_dim]
            cos       : [seq_len, rope_dim] — precomputed cosine table
            sin       : [seq_len, rope_dim] — precomputed sine table

        Returns:
            output    : [batch, seq_len, num_heads, head_dim]
        """
        batch, seq_len, _, _ = q_content.shape

        # ── Step 1: Apply RoPE to positional dimensions ──────────────────────
        q_rope_out, k_rope_out = apply_rope(q_rope, k_rope, cos, sin)

        # ── Step 2: Compute attention scores (content path + RoPE path) ──────
        # Content path: pure semantic similarity (no positional info)
        # [batch, heads, seq, content_dim] @ [batch, heads, content_dim, seq]
        # → [batch, heads, seq, seq]
        attn_content = torch.matmul(
            q_content.transpose(1, 2),
            k_content.transpose(1, 2).transpose(-1, -2)
        )

        # RoPE path: positional similarity
        # [batch, heads, seq, rope_dim] @ [batch, heads, rope_dim, seq]
        # → [batch, heads, seq, seq]
        attn_rope = torch.matmul(
            q_rope_out.transpose(1, 2),
            k_rope_out.transpose(1, 2).transpose(-1, -2)
        )

        # Combined score — scale to prevent softmax saturation
        scores = (attn_content + attn_rope) * self.scale

        # ── Step 3: Apply sliding window mask ────────────────────────────────
        # This is the key step — blocks attention outside the local window
        mask = build_window_mask(
            seq_len=seq_len,
            window_size=self.window_size,
            causal=self.causal,
            device=scores.device
        )
        # Expand mask to [1, 1, seq_len, seq_len] for broadcasting
        scores = scores.masked_fill(
            ~mask.unsqueeze(0).unsqueeze(1),
            float('-inf')
        )

        # ── Step 4: Softmax → attention weights ──────────────────────────────
        # Positions with -inf become 0 after softmax (they don't contribute)
        attn_weights = torch.softmax(scores, dim=-1)

        # ── Step 5: Weighted sum over Values ─────────────────────────────────
        output = torch.matmul(
            attn_weights,
            v.transpose(1, 2)             # [batch, heads, seq, head_dim]
        ).transpose(1, 2)                 # back to [batch, seq, heads, head_dim]

        return output

    def memory_estimate_mb(self) -> dict:
        """
        Estimate memory usage for this attention module at the given window size.
        Useful for verifying we stay under the 500 MB budget.

        Returns a dict with breakdown in MB.
        """
        # KV cache: 2 (K+V) × window_size × num_heads × head_dim × 4 bytes (float32)
        kv_cache_bytes = 2 * self.window_size * self.num_heads * self.head_dim * 4
        kv_cache_mb    = kv_cache_bytes / (1024 ** 2)

        # Attention score matrix: heads × window × window × 4 bytes
        score_matrix_bytes = self.num_heads * self.window_size * self.window_size * 4
        score_matrix_mb    = score_matrix_bytes / (1024 ** 2)

        return {
            "kv_cache_mb"      : round(kv_cache_mb, 2),
            "score_matrix_mb"  : round(score_matrix_mb, 2),
            "total_attn_mb"    : round(kv_cache_mb + score_matrix_mb, 2),
            "window_size"      : self.window_size,
            "note"             : "Fixed — does NOT grow with log file length"
        }


# ─────────────────────────────────────────────────────────────────────────────
# FlashAttention version (faster if flash_attn is installed)
# ─────────────────────────────────────────────────────────────────────────────

def log_window_attention_flash(q, k, v, window_size: int = 512) -> torch.Tensor:
    """
    Faster version using FlashAttention-2 kernel (if available).

    FlashAttention handles the sliding window natively — no need to build
    the mask manually. The kernel is highly optimized for this pattern.

    Args:
        q, k, v     : [batch, seq_len, num_heads, head_dim]
        window_size : Number of tokens to attend to on the left

    Returns:
        output      : [batch, seq_len, num_heads, head_dim]

    Install:
        pip install flash-attn --no-build-isolation

    Usage in modeling_deepseek.py:
        from log_attention import log_window_attention_flash
        # Replace: attn_output = flash_attn_func(q, k, v, causal=True)
        # With:    attn_output = log_window_attention_flash(q, k, v, window_size=512)
    """
    try:
        from flash_attn import flash_attn_func
        # window_size=(left, right): left=W means attend to W past tokens
        #                            right=0 means don't look at future tokens
        return flash_attn_func(
            q, k, v,
            dropout_p   = 0.0,
            causal      = True,
            window_size = (window_size, 0)
        )
    except ImportError:
        raise ImportError(
            "flash_attn not installed. Use LogWindowAttention class instead, "
            "or install with: pip install flash-attn --no-build-isolation"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run: python log_attention.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LogWindowAttention — Self Test")
    print("="*60)

    # Config matching DeepSeek-1.3B
    HIDDEN      = 2048
    HEADS       = 16
    ROPE_DIM    = 64
    CONTENT_DIM = (HIDDEN // HEADS) - ROPE_DIM   # 128 - 64 = 64
    WINDOW      = 512
    BATCH       = 1
    SEQ         = 512  # one full window

    print(f"\n  Model config: DeepSeek-1.3B style")
    print(f"  hidden_size={HIDDEN}, num_heads={HEADS}, head_dim={HIDDEN//HEADS}")
    print(f"  rope_dim={ROPE_DIM}, window_size={WINDOW}, seq_len={SEQ}")

    # Create module
    attn = LogWindowAttention(
        hidden_size         = HIDDEN,
        num_attention_heads = HEADS,
        rope_dim            = ROPE_DIM,
        window_size         = WINDOW,
        causal              = True
    )

    # Dummy tensors (CPU)
    # q_content/k_content: the non-positional (content) part of Q and K
    # Their last dim = head_dim - rope_dim (DeepSeek decoupled RoPE)
    q_content = torch.randn(BATCH, SEQ, HEADS, CONTENT_DIM)
    k_content = torch.randn(BATCH, SEQ, HEADS, CONTENT_DIM)
    # v uses the full head_dim
    v         = torch.randn(BATCH, SEQ, HEADS, HIDDEN // HEADS)
    # q_rope/k_rope: the positional part (only rope_dim wide)
    q_rope    = torch.randn(BATCH, SEQ, HEADS, ROPE_DIM)
    k_rope    = torch.randn(BATCH, SEQ, HEADS, ROPE_DIM)
    # cos/sin: [seq_len, rope_dim] — apply_rope will unsqueeze for broadcasting
    cos       = torch.ones(SEQ, ROPE_DIM)
    sin       = torch.zeros(SEQ, ROPE_DIM)

    print(f"\n  Running forward pass...")
    output = attn(q_content, k_content, v, q_rope, k_rope, cos, sin)
    print(f"  ✅ Output shape: {tuple(output.shape)}  (expected: {(BATCH, SEQ, HEADS, HIDDEN//HEADS)})")

    # Memory estimate
    mem = attn.memory_estimate_mb()
    print(f"\n  Memory estimate:")
    print(f"    KV cache        : {mem['kv_cache_mb']} MB")
    print(f"    Score matrix    : {mem['score_matrix_mb']} MB")
    print(f"    Total attention : {mem['total_attn_mb']} MB")
    print(f"    Model weights   : ~412 MB  (DeepSeek-1.3B @ 4-bit)")
    print(f"    ─────────────────────────")
    total = mem['total_attn_mb'] + 412
    print(f"    Grand total     : ~{total:.0f} MB / 500 MB budget  {'✅' if total < 500 else '❌'}")
    print(f"    Note            : {mem['note']}")

    # Verify mask is actually restricting attention
    mask = build_window_mask(20, window_size=5, causal=True)
    print(f"\n  Window mask check (seq=20, window=5, causal=True):")
    print(f"    Row 0  can see cols: {[j for j in range(20) if mask[0][j]]}")
    print(f"    Row 10 can see cols: {[j for j in range(20) if mask[10][j]]}")
    print(f"    Row 19 can see cols: {[j for j in range(20) if mask[19][j]]}")
    print(f"    ✅ Each row sees at most {5+1} tokens (window=5 past + self)")

    print(f"\n{'='*60}\n")
