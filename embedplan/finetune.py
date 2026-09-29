"""Unfreezing the encoder.

The paper's claim is about *frozen* embeddings. One explanation for the
extrapolation failure is that those embeddings cluster by problem instance rather
than by structural role. Without an intervention that explanation is only
inferred from the fact that bigger encoders help. Fine-tuning tests it
directly: if the clustering is what limits extrapolation, adapting the encoder
should move it; if extrapolation stays flat, the limitation is elsewhere.

Positioned as a diagnostic upper bound ("what does the frozen constraint cost?"),
not as a change of method.

Only long-context encoders are used here. all-mpnet-base-v2 truncates at 384
tokens against state prompts averaging ~500 (ferry p95 = 1019), which would
confound fine-tuning with simply seeing more of the input.

The encoder is adapted with LoRA rather than fully fine-tuned: it keeps the
frozen backbone intact, fits alongside the projection heads and transition net in
one optimizer, and makes the 7B arm affordable.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from embedplan.losses import (compute_action_loss, compute_delta_action_loss,
                              compute_delta_infonce_loss, compute_infonce_loss)

# LoRA target modules by architecture family. peft needs exact submodule names.
LORA_TARGETS = {
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "xlm-roberta": ["query", "key", "value", "dense"],
    "bert": ["query", "key", "value", "dense"],
    "mpnet": ["q", "k", "v", "o"],
}

# Causal LMs have no [CLS]; pool the last non-pad position instead.
DEFAULT_POOLING = {"qwen2": "last", "llama": "last", "mistral": "last",
                   "xlm-roberta": "cls", "bert": "mean", "mpnet": "mean"}


class TextBank:
    """Maps embedding-table rows back to the text they were encoded from.

    The cached pipeline stores prompt -> row; fine-tuning needs row -> prompt so a
    batch of triplet indices can be re-tokenized on the fly.
    """

    def __init__(self, ds):
        n_states = ds.state_embs.shape[0]
        self.state_texts: List[Optional[str]] = [None] * n_states
        for prompt, row in ds.state_prompt_map.items():
            if 0 <= row < n_states:
                self.state_texts[row] = prompt
        missing = sum(t is None for t in self.state_texts)
        if missing:
            raise ValueError(f"{missing}/{n_states} state rows have no prompt text; "
                             "the embedding cache and the factorized data disagree")
        self.action_texts: List[str] = list(ds.action_vocab)

    def states(self, idx) -> List[str]:
        return [self.state_texts[int(i)] for i in idx]

    def actions(self, idx) -> List[str]:
        return [self.action_texts[int(i)] for i in idx]


def _family(model_name: str, config) -> str:
    mt = (getattr(config, "model_type", "") or "").lower()
    for key in LORA_TARGETS:
        if key in mt or key in model_name.lower():
            return key
    return mt or "bert"


class TunableEncoder(nn.Module):
    """A HF encoder with pooling, optionally wrapped in LoRA adapters.

    `lora_rank=0` leaves the backbone frozen, which is how the control arm runs —
    the same code path as the tuned arm so the two are compared like for like.
    """

    def __init__(self, model_name: str, lora_rank: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, pooling: Optional[str] = None,
                 max_length: int = 1024, dtype=torch.bfloat16, device="cuda",
                 gradient_checkpointing: bool = True):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.device_ = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=dtype)

        fam = _family(model_name, self.backbone.config)
        self.pooling = pooling or DEFAULT_POOLING.get(fam, "mean")
        self.family = fam

        if gradient_checkpointing and lora_rank > 0:
            self.backbone.gradient_checkpointing_enable()
            self.backbone.enable_input_require_grads()

        if lora_rank > 0:
            from peft import LoraConfig, get_peft_model
            targets = [t for t in LORA_TARGETS.get(fam, ["query", "key", "value"])
                       if any(t == n.split(".")[-1] for n, _ in self.backbone.named_modules())]
            if not targets:
                raise ValueError(f"no LoRA target modules matched for family '{fam}'")
            self.backbone = get_peft_model(self.backbone, LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=targets, bias="none"))
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            print(f"[encoder] {model_name} family={fam} pooling={self.pooling} "
                  f"LoRA r={lora_rank} on {targets} -> {trainable:,} trainable", flush=True)
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            print(f"[encoder] {model_name} family={fam} pooling={self.pooling} FROZEN", flush=True)

        self.backbone.to(device)
        self.hidden_size = self.backbone.config.hidden_size

    def _pool(self, hidden, mask):
        if self.pooling == "cls":
            return hidden[:, 0]
        if self.pooling == "last":
            last = mask.sum(1) - 1
            return hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        batch = self.tokenizer(list(texts), padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device_)
        out = self.backbone(**batch, return_dict=True)
        return self._pool(out.last_hidden_state, batch["attention_mask"]).float()

    @torch.no_grad()
    def encode_all(self, texts: Sequence[str], batch_size: int = 16,
                   desc: str = "encoding") -> torch.Tensor:
        """Re-encode a whole state pool for evaluation. Sorted by length so padding
        waste stays low, then restored to the original order."""
        was_training = self.training
        self.eval()
        order = np.argsort([len(t) for t in texts])
        out = torch.empty(len(texts), self.hidden_size, dtype=torch.float32, device=self.device_)
        for i in tqdm(range(0, len(order), batch_size), desc=desc, leave=False):
            sel = order[i:i + batch_size]
            out[torch.as_tensor(sel, device=self.device_)] = self([texts[j] for j in sel])
        if was_training:
            self.train()
        return out


class JointModel(nn.Module):
    """Encoder + projection heads + transition network, trained end to end."""

    def __init__(self, encoder: TunableEncoder, state_projection_head, action_projection_head,
                 transition_model):
        super().__init__()
        self.encoder = encoder
        self.state_projection_head = state_projection_head
        self.action_projection_head = action_projection_head
        self.transition_model = transition_model

    def forward(self, state_texts, action_texts):
        s = self.state_projection_head(self.encoder(state_texts))
        a = self.action_projection_head(self.encoder(action_texts))
        return self.transition_model(s, a)


def train_joint(model: JointModel, bank: TextBank, tri, train_idx, args, device,
                log_every: int = 20):
    """Contrastive training with the encoder in the loop.

    Batches are far smaller than the frozen pipeline's (encoding dominates), so
    in-batch negatives are correspondingly fewer. Under the extrapolation split
    batches are still drawn within a problem, keeping negatives hard.
    """
    s_i = tri["s_emb_idx"].to_numpy()[train_idx]
    a_i = tri["a_idx"].to_numpy()[train_idx]
    p_i = tri["sp_emb_idx"].to_numpy()[train_idx]
    prob = tri["problem_idx"].to_numpy()[train_idx]

    groups = None
    if getattr(args, "split", "random") != "random":
        by_prob: Dict[int, List[int]] = {}
        for pos, pid in enumerate(prob):
            by_prob.setdefault(int(pid), []).append(pos)
        groups = [np.array(v) for v in by_prob.values()]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("encoder")], "lr": args.encoder_lr},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("encoder")], "lr": args.lr},
    ])
    print(f"[train] {sum(p.numel() for p in params):,} trainable params", flush=True)

    rng = np.random.default_rng(args.seed)
    n, B = len(train_idx), args.batch_size
    step = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        if groups is None:
            perm = rng.permutation(n)
            batches = [perm[i:i + B] for i in range(0, n, B)]
        else:
            batches = []
            for g in groups:
                gg = g.copy()
                rng.shuffle(gg)
                batches.extend([gg[i:i + B] for i in range(0, len(gg), B)])
            rng.shuffle(batches)

        total, seen = 0.0, 0
        for b in tqdm(batches, desc=f"epoch {ep}", leave=False):
            if len(b) < 2:
                continue
            s_txt = bank.states(s_i[b])
            a_txt = bank.actions(a_i[b])
            p_txt = bank.states(p_i[b])

            # one encoder pass over states and next-states keeps activations shared
            both = model.encoder(s_txt + p_txt)
            s_emb, sp_emb = both[:len(b)], both[len(b):]
            a_emb = model.encoder(a_txt)

            s_proj = model.state_projection_head(s_emb)
            a_proj = model.action_projection_head(a_emb)
            sp_proj = model.state_projection_head(sp_emb)
            pred = model.transition_model(s_proj, a_proj)

            # The encoder is only shaped by whatever geometry the loss lives in, so the
            # joint stage has to use the same one as the head stage and the scorer —
            # otherwise LoRA optimizes absolute positions that DELTA scoring ignores.
            delta = getattr(args, "loss_space", "absolute") == "delta"
            if delta:
                loss = compute_delta_infonce_loss(pred, sp_proj, s_proj, args.tau)
            else:
                loss = compute_infonce_loss(pred, sp_proj, args.tau)
            if args.action_weight > 0:
                m = min(max(2, int(len(b) ** 0.5)), len(b))
                if delta:
                    loss = loss + args.action_weight * compute_delta_action_loss(
                        _TransitionOnly(model), s_proj, a_proj, sp_proj, s_proj,
                        args.tau, m, device)
                else:
                    loss = loss + args.action_weight * compute_action_loss(
                        _TransitionOnly(model), s_proj, a_proj, sp_proj, args.tau, m, device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()

            total += loss.item() * len(b)
            seen += len(b)
            step += 1
            if step % log_every == 0:
                print(f"  ep {ep} step {step:5d}  loss {total / max(1, seen):.4f}", flush=True)

        print(f"[train] epoch {ep} mean loss {total / max(1, seen):.4f}", flush=True)
    return model


class _TransitionOnly(nn.Module):
    """Adapter so compute_action_loss can call the transition net on already-projected
    inputs, matching how it is used in the frozen pipeline."""

    def __init__(self, joint: JointModel):
        super().__init__()
        self.joint = joint
        self.state_projection_head = nn.Identity()

    def forward(self, s_proj, a_proj):
        return self.joint.transition_model(s_proj, a_proj)
