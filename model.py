#model and trainin
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from pathlib import Path
from data import ProteinCache, get_train_batch, get_eval_batch
from preprocess import preprocess_fasta

#approx size of attention param (q/k/v/out param) ~4 * d_model^2
#approx size of swiglu params (3* d_model * d_ff) --> 12 * d_model^2
#d_ff = 4 * d_model
#approximate model size is 16 x n_layers x d_model^2
#this one is about 16.7 million params
BATCH_SIZE = 512 # gonna change later
VOCAB_SIZE = 24
CONTEXT_LENGTH = 512
D_MODEL = 256
N_LAYERS = 16
N_HEADS = 4
D_FF = 4 * D_MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"

# x = [B, T, d_model]
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        output = x / rms * self.scale
        return output

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # gate_proj shape == up_proj shape :   [B, T, d_ff]
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        # down_proj: [B, T, d_ff] -> [B, T, d_model]
        #down_proj maps hidden FFN dimension back to model dimension
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        #F.silu ===> gate * sigmoid(gate)
        hidden = F.silu(gate) * up
        output = self.down_proj(hidden)

        return output

#very similar to https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/train_gpt_simple.py
#only big difference is the arrangement of the shape
#paper -----> https://arxiv.org/pdf/2104.09864
class Rotary_embed(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        assert head_dim % 2 ==0

        angular_freq = (1/1024) ** torch.linspace(
            0, 1, steps=head_dim//4, dtype=torch.float32
            )
        self.register_buffer(
            name="angular_freq",
            tensor=torch.cat([angular_freq, angular_freq.new_zeros(head_dim//4)])
            )

    def forward(self, x):
        #x.size(2) not 1, we want for sequence lenght T, x shape: [B, H, T, head_dim]
        #angular_freq has shape [16] + [16] = [32], 16 is from 64//4
        pos = torch.arange(x.size(2), dtype=torch.float32, device=x.device)
        theta = torch.outer(pos, self.angular_freq)[None, None,:, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=-1).type_as(x)

import numpy as np
class BidiAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rotary = Rotary_embed(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, pad_mask=None):
        B, T, D = x.shape

        #projection of q, k and v
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        #[8, 512, 256] ===> [8, 512, 4, 64] ===> [8, 4, 512, 64]
        #[B, T,   D]   ========================> [B, N_HEAD, T, D_MODEL]
        #attention is computed separately per head so typical
        #arrangement is like
        # batch 0, head 0, all tokens, head_dim
        #batch 0, head 1, all tokens, head_dim
        #batch 0, head 2, all tokens, head_dim
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1,2)

        q = self.rotary(q)
        k = self.rotary(k)

        #q:                 [B, H, T, Dh]
        #k.transpose(-2,-1):[B, H, Dh, T]
        #scores:            [B, H, T, T]
        #scores =torch.einsum("bhtd,bhsd->bhts", q, k)
        #one scores tensor is about 2GB in float 32 ==> 512 × 4 × 512 × 512
        #536,870,912 × 4 bytes ==> 2.0 GiB
        #so we get an out of memory error on an A100, unless we use optimised attention
        #scores = q @ k.transpose(-2, -1)
        #divide by head_dim not d_model
        #scores = scores / (self.head_dim ** 0.5)
        attn_mask = None
        if pad_mask is not None:
            #[B, H, T_query, T_key] =====> [B, 1, 1, T_key]
            attn_mask = pad_mask[:, None, None, :]
            #scores = scores.masked_fill(key_mask== 0, float("-inf"))

        #weights = torch.softmax(scores, dim=-1)
        #out: [B, H, T, Dh]
        #out = torch.einsum("bhts,bhsd->bhtd", weights, v)
        #out = weights @ v
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )
        out = out.transpose(1,2).contiguous().view(B,T,D)
        #out = [B, T, D]
        out = self.out_proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attn = BidiAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.norm1(x), pad_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class ProteinMLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, n_layers, n_heads, d_ff):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, pad_mask=None):
        x = self.token_emb(input_ids)

        for block in self.blocks:
            x = block(x, pad_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

processed_dir = Path("dataset_PLM/processed")

required_cache_files = [
    processed_dir / "tokens.bin",
    processed_dir / "offsets.npy",
    processed_dir / "lengths.npy",
    processed_dir / "train_indices.npy",
    processed_dir / "eval_indices.npy",
    processed_dir / "meta.json",
]

cache_exists = all(path.exists() for path in required_cache_files)

if not cache_exists:
    preprocess_fasta(max_records=10000)
else:
    print("found processed cache")

cache = ProteinCache()
print("cache loaded")

model = ProteinMLM(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF,
).to(device)

#comment out if not needed
num_params= sum(p.numel() for p in model.parameters())
print(f"parameters--> {num_params:,}")

def split_params_for_muon(model):
    muon_params = []
    adamw_params = []

    muon_names = []
    adamw_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_hidden_matrix = name.startswith("blocks.") and param.ndim == 2

        if is_hidden_matrix:
            muon_params.append(param)
            muon_names.append(name)
        else:
            adamw_params.append(param)
            adamw_names.append(name)

    return muon_params, adamw_params, muon_names, adamw_names

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
muon_params, adamw_params, muon_names, adamw_names = split_params_for_muon(model)

muon_optim = torch.optim.Muon(
    muon_params,
    lr=1e-3,
    weight_decay=0.01,
    ns_steps=5,
)

adamw_optim = torch.optim.AdamW(
    adamw_params,
    lr=1e-4,
    weight_decay=0.01,
)
model = torch.compile(model)

def masked_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    mask = labels != -100

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    return (preds[mask] == labels[mask]).float().mean()

num_steps = 3500
log_every = 100
max_grad_norm = 1.0
eval_every = 250

#late stage eval is for when the eval loss gets below 2.0, so we can track
#more closely at the later stage. Our goal is to reduce the amount of time
#before we get to 1.86 loss (similar to modded-nanogpt)
late_stage_eval_every = 50
late_stage_eval_threshold = 2.0

target_eval_loss = 1.86
num_eval_batches = 5

model.train()

start_time = time.time()

from tqdm.auto import tqdm

pbar = tqdm(range(1, num_steps + 1), desc="training")

for step in pbar:
    batch = get_train_batch(cache, batch_size=BATCH_SIZE, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(batch["input_ids"], batch["pad_mask"])

        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            batch["labels"].reshape(-1),
            ignore_index=-100
    )

    muon_optim.zero_grad(set_to_none=True)
    adamw_optim.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    muon_optim.step()
    adamw_optim.step()

    if step % log_every == 0 or step == 1:
        acc = masked_accuracy(logits, batch["labels"])
        elapsed = time.time() - start_time

        log_msg = (
            f"step {step:5d} |"
            f"loss {loss.item():.4f} |"
            f"acc {acc.item():.4f} |"
            f"time {elapsed:.1f}s"
        )
        pbar.write(log_msg)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc.item():.4f}",
            "time": f"{elapsed:.1f}s",

        })

    if step % eval_every == 0:
        model.eval()

        eval_loss_total = 0.0
        eval_acc_total = 0.0

        with torch.no_grad():
            for eval_idx in range(num_eval_batches):
                start = eval_idx * BATCH_SIZE

                eval_batch = get_eval_batch(cache=cache,
                                            start=start,
                                            batch_size= BATCH_SIZE,
                                            device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    eval_logits = model(eval_batch["input_ids"], eval_batch["pad_mask"])

                    eval_loss = F.cross_entropy(
                        eval_logits.reshape(-1, VOCAB_SIZE),
                        eval_batch["labels"].reshape(-1),
                        ignore_index=-100,
                    )

                eval_acc = masked_accuracy(eval_logits, eval_batch["labels"])

                eval_loss_total += eval_loss.item()
                eval_acc_total += eval_acc.item()

        eval_loss_avg = eval_loss_total / num_eval_batches
        eval_acc_avg = eval_acc_total / num_eval_batches

        pbar.write(
            f"\n=====>EVAL step {step:5d} | "
            f"eval_loss {eval_loss_avg:.4f} | "
            f"eval_acc {eval_acc_avg:.4f}<======\n"
        )

        if eval_loss_avg <= target_eval_loss:
            elapsed = time.time() - start_time

            pbar.write(
                f"TARGET REACHED | step {step} | "
                f"eval_loss {eval_loss_avg:.4f} | "
                f"time {elapsed:.1f}s"
            )
            break

        if eval_loss_avg < late_stage_eval_threshold and eval_every != late_stage_eval_every:
            old_eval_every = eval_every
            eval_every = late_stage_eval_every
            pbar.write(f"eval changed from {old_eval_every} to {eval_every}")

        model.train()


torch.save(
    {
        "model": model._orig_mod.state_dict(),
        "config": {
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "d_ff": D_FF
        },
    },
    "protein_mlm(withROPEEEE).pt",
)