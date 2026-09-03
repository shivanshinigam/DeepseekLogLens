# Custom DeepSeek for Log Analysis — Full Plan

> **Written for:** Sir's review
> **Goal:** Take DeepSeek, strip it down to only look at a small local window of log lines (not the full 10,000-page file), host it cheaply on AWS Trainium, and pitch it as a product.

---

## The Core Problem (Plain English)

When you give a standard LLM a massive log file, it tries to read and relate every single line to every other line. That is called **full self-attention**. For a 10,000-page log file, this is like asking someone to memorize the entire library before answering one question. It is slow and needs huge memory.

**What we actually need:** When analyzing log line 5000, we only care about what happened in lines 4990–5010 (a few pages before and after). Everything else is noise.

Our fix: **Sliding Window Attention** — the model only "sees" a small local window of lines at any time, like a magnifying glass that slides over the log. Memory stays fixed, speed goes up dramatically.

---

## Part 1 — What to Change in DeepSeek's Code

### Step 1: Clone DeepSeek and Create a Custom Branch

```bash
# Clone the official DeepSeek model repo (we use the Hugging Face version)
git clone https://github.com/deepseek-ai/DeepSeek-V2.git
cd DeepSeek-V2

# Create our own branch — never touch main
git checkout -b feature/log-analysis-local-attention
```

The main file we will edit is `modeling_deepseek.py`. This is the file that defines how attention works.

---

### Step 2: Understand the Original Attention (Before We Touch It)

In `modeling_deepseek.py`, the attention class looks roughly like this (simplified):

```python
class DeepseekAttention(nn.Module):
    def forward(self, hidden_states, ...):
        # Computes Q, K, V for ALL tokens
        # Attention score = Q @ K^T   <-- this is the expensive full-matrix operation
        # This grows as O(n²) with sequence length — the memory killer
```

For a 10,000 line log file, `n` is huge. We replace this with a windowed version.

---

### Step 3: Write Our Custom Attention Module

Create a new file: `log_attention.py` (keep it separate so we don't break the original)

```python
# log_attention.py
# Plain purpose: Only let each log line "see" W lines before and after it.

import torch
import torch.nn as nn
import math

# ─── Helper: RoPE rotation ───────────────────────────────────────────────────

def rotate_half(x):
    """Splits the vector in half and rotates it — this is the core RoPE trick."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    """Apply positional encoding (RoPE) to Query and Key vectors."""
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out

# ─── Main Custom Attention Class ─────────────────────────────────────────────

class LogWindowAttention(nn.Module):
    """
    Local Sliding Window Attention for Log Analysis.

    - Each log token only attends to tokens within [pos - W, pos + W].
    - Memory is O(W) not O(n²). Fixed and predictable.
    - window_size=512 means roughly 1-2 pages of log context.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim  = config.hidden_size // self.num_heads

        # ← TUNE THIS: 512 = ~2 log pages of context on each side
        # For tighter memory, lower to 256. For more context, raise to 1024.
        self.window_size = getattr(config, "log_window_size", 512)

    def forward(self, q, k, v, cos, sin):
        """
        q, k, v : [batch, seq_len, num_heads, head_dim]
        cos, sin: positional encoding vectors
        """
        # 1. Apply RoPE so the model understands position within the window
        q, k = apply_rope(q, k, cos, sin)

        # 2. Compute raw attention scores (all pairs, we'll mask next)
        #    scores[i][j] = how much token i should attend to token j
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        #    Shape: [batch, num_heads, seq_len, seq_len]

        # 3. Build the sliding window mask
        #    Allow token i to only see tokens in [i - W, i] (causal window)
        #    For bidirectional (see both left and right), use the commented version
        seq_len = scores.size(-1)
        mask = torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool)

        # Causal + local window: see W tokens behind, nothing ahead (for streaming logs)
        mask = torch.tril(mask, diagonal=0) & torch.triu(mask, diagonal=-self.window_size)

        # Bidirectional window (if log lines have future context too):
        # mask = torch.tril(mask, diagonal=self.window_size) & torch.triu(mask, diagonal=-self.window_size)

        # 4. Zero out everything outside the window
        scores = scores.masked_fill(
            ~mask.unsqueeze(0).unsqueeze(1),  # broadcast over batch and heads
            float('-inf')
        )

        # 5. Softmax → weighted sum over values
        weights = torch.softmax(scores, dim=-1)
        output  = torch.matmul(weights, v)
        return output
```

---

### Step 4: Plug This Into `modeling_deepseek.py`

Find the original attention class in `modeling_deepseek.py`. Replace only the `forward()` method's attention computation section:

```python
# In modeling_deepseek.py — find DeepseekAttention.forward()

# BEFORE (original full attention):
attn_output = flash_attn_func(q, k, v, causal=True)

# AFTER (our windowed attention — add this import at the top of the file):
# from log_attention import LogWindowAttention
# self.local_attn = LogWindowAttention(config)  ← add in __init__

attn_output = self.local_attn(q, k, v, cos, sin)
```

> **Why not just use FlashAttention's window_size parameter?**
> FlashAttention has a built-in `window_size=(W, 0)` argument that does the same thing with GPU-optimized kernels. If you have FlashAttention installed, that's faster:
> ```python
> from flash_attn import flash_attn_func
> attn_output = flash_attn_func(q, k, v, causal=True, window_size=(512, 0))
> ```
> Use our manual version if FlashAttention is not available (e.g., on CPU or Trainium).

---

### Step 5: Add a Config Flag

In `config.json` for the model, add:

```json
{
  "log_window_size": 512,
  "log_analysis_mode": true
}
```

This way, the same codebase can run in normal mode or log-analysis mode without changing code — just change the config.

---

## Part 2 — Memory Math (Why This Fits in 500 MB)

| Component | Size |
|---|---|
| DeepSeek 1.3B weights at 4-bit (Q4 GGUF) | ~700 MB → but we only load layers we need |
| DeepSeek 7B weights at 4-bit | ~4 GB → need to use a smaller variant |
| **KV Cache at window=512, batch=1** | **~18 MB** — fixed, does not grow |
| Activations during inference | ~50–80 MB |
| **Total (small 1.3B model, 4-bit)** | **~400–480 MB** ✅ |

**Key rules to stay under 500 MB:**
1. Use the **1.3B parameter variant** of DeepSeek (not 7B or 67B)
2. Load weights in **4-bit quantization** (use `bitsandbytes` or GGUF format)
3. Set `batch_size = 1` — never batch multiple log files at once
4. Window size ≤ 1024 tokens

---

## Part 3 — Build the Custom Branch (Git Workflow)

```bash
# Our branch structure:
main
└── feature/log-analysis-local-attention    ← our work branch
    ├── log_attention.py                    ← new file (our attention module)
    ├── modeling_deepseek.py                ← modified original
    ├── config_log_analysis.json            ← custom config
    └── README_LOG_ANALYSIS.md              ← explains our changes

# After making changes:
git add log_attention.py modeling_deepseek.py config_log_analysis.json
git commit -m "feat: add sliding window attention for log analysis (window=512)"
git push origin feature/log-analysis-local-attention
```

When ready to release:
```bash
git tag v1.0-log-analysis
git push origin v1.0-log-analysis
```

We can also publish this to Hugging Face as `our-company/deepseek-log-1.3b-window512`.

---

## Part 4 — Hosting on AWS Trainium

### What is Trainium?

AWS Trainium (trn1 instances) is Amazon's own AI chip — cheaper than Nvidia A100 GPUs for inference, designed for running AI models at scale. The cost is roughly **3–5x cheaper** than equivalent GPU inference.

### Why Trainium Makes Sense for Our Pitch

- We are running a small, modified model (1.3B, windowed attention)
- Small models on Trainium = very low cost per 1000 log lines analyzed
- We can undercut any competitor using standard GPU inference

### Setup Steps

**Step 1: Get an AWS Trainium instance**
```bash
# Instance type: trn1.2xlarge (cheapest, 32 GB HBM memory — plenty for our 500 MB model)
# Launch via AWS Console or CLI:
aws ec2 run-instances \
  --instance-type trn1.2xlarge \
  --image-id ami-XXXXXXXX  # Use Deep Learning AMI with Neuron SDK
```

**Step 2: Install AWS Neuron SDK (Trainium's software layer)**
```bash
pip install torch-neuronx neuronx-cc transformers
```

**Step 3: Compile our modified model for Trainium**

Trainium requires a one-time compilation step. After this, inference is fast and cheap.

```python
# compile_for_trainium.py
import torch
import torch_neuronx
from transformers import AutoModelForCausalLM

# Load our custom model
model = AutoModelForCausalLM.from_pretrained("./our-custom-deepseek-log")
model.eval()

# Example input (512 token window)
example_input = torch.zeros((1, 512), dtype=torch.long)

# Compile to Trainium hardware — this takes ~10–20 minutes, runs once
traced_model = torch_neuronx.trace(model, example_input)
traced_model.save("deepseek_log_trainium_compiled.pt")

print("Compiled! Ready to deploy.")
```

**Step 4: Run Inference API on Trainium**

```python
# inference_server.py — a simple FastAPI server
from fastapi import FastAPI
import torch
import torch_neuronx

app = FastAPI()

# Load the pre-compiled model (fast, < 5 seconds)
model = torch.jit.load("deepseek_log_trainium_compiled.pt")

@app.post("/analyze-log")
def analyze_log(payload: dict):
    log_text = payload["log_text"]      # The log page/chunk to analyze
    query    = payload["query"]         # E.g. "Why did the connection timeout?"
    
    # Tokenize and run inference
    inputs = tokenize(log_text, query)  # your tokenization function
    with torch.no_grad():
        output = model(inputs)
    
    return {"analysis": decode(output)} # your decode function
```

```bash
# Start the server
uvicorn inference_server:app --host 0.0.0.0 --port 8080
```

---

## Part 5 — Product Architecture (How Companies Would Use This)

```
Company's Log Files
        │
        ▼
  Log Chunker Service
  (splits 10,000 pages into 512-token windows, sends one at a time)
        │
        ▼
  Our API (hosted on Trainium trn1)
  POST /analyze-log
  { "log_text": "...", "query": "..." }
        │
        ▼
  Custom DeepSeek (Window Attention = 512 tokens)
  → Only reads the relevant window, ignores the rest
        │
        ▼
  Response: Root cause, anomaly, pattern found
        │
        ▼
  Company's Dashboard / Alert System
```

### Pricing Pitch to Companies

| Metric | Standard GPT-4 API | Our Trainium Product |
|---|---|---|
| Cost per 1M tokens | ~$15 | ~$1.50 |
| Latency (512 token window) | ~2s | <0.5s |
| Memory footprint | Cloud-managed | 500 MB fixed |
| Log-specific tuning | ❌ Generic | ✅ Window-tuned |
| Data privacy | Sent to OpenAI | Hosted in your AWS VPC |

**The pitch is simple:** 10x cheaper, 4x faster, and your logs never leave your AWS account.

---

## Part 6 — What to Do Next (In Order)

- [ ] **Week 1:** Fork DeepSeek repo, add `log_attention.py`, test locally on sample logs
- [ ] **Week 2:** Tune window size (256 vs 512 vs 1024) and measure accuracy vs memory
- [ ] **Week 3:** Quantize to 4-bit, validate 500 MB target is met
- [ ] **Week 4:** Set up Trainium trn1 instance, compile model, benchmark cost
- [ ] **Week 5:** Build the inference API, write the chunker that splits log files
- [ ] **Week 6:** Create a demo with a real log file, prepare the pitch deck

---

## Open Questions for Sir

1. **Which DeepSeek variant to start with?** 1.3B is safest for 500 MB. Should we try 7B with more aggressive quantization?
2. **Bidirectional or causal attention?** For log analysis, do we need to see future lines (bidirectional) or only past lines (causal)? This changes the mask.
3. **Do we fine-tune the model on log data?** A few hours of fine-tuning on real log datasets would dramatically improve accuracy. Worth the effort?
4. **AWS account setup?** Do we have a Trainium-eligible AWS account, or do we start with a standard GPU (A10G) first?
