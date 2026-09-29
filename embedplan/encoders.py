"""Frozen text encoders that turn state and action descriptions into embeddings.

Llama3ModelEncoder mean-pools a Hugging Face model's last hidden layer (used for Llama-3.3-70B and
Qwen2.5-7B). SentenceTransformerEncoder wraps sentence-transformers models (MPNet, BGE-M3).
HashingEncoder hashes word n-grams and needs no download, which makes it a quick encoder for
trying the method and for tests. `get_encoder` turns any of these, a model name, or your own
function into one callable: a list of texts in, a float32 array of shape (n_texts, dim) out.
"""

from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch


class HFEncoderBase:
    def __init__(self, model_name: str, pooling_strategy: str = "last", device: Optional[str] = None,
                 torch_dtype: torch.dtype = torch.bfloat16, attn_impl: Optional[str] = None):
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.device = device or ('cuda' if torch.cuda.is_available() else
                                 ('mps' if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else 'cpu'))

        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # the `encoders` extra

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        common = dict(trust_remote_code=True, device_map="auto", torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        if attn_impl:
            common["attn_implementation"] = attn_impl

        try:
            self.model = AutoModel.from_pretrained(model_name, **common)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **common)

        self.model.eval()
        self._is_sharded = hasattr(self.model, "hf_device_map") and len(set(map(str, self.model.hf_device_map.values()))) > 1
        self._input_device = "cpu" if self._is_sharded else self.device
        self.max_len = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if not isinstance(self.max_len, int) or self.max_len > 10**8:
            self.max_len = min(getattr(self.tokenizer, "model_max_length", 4096), 32768)
        print(f"[HFEncoderBase] {model_name} loaded | sharded={self._is_sharded} | dtype={torch_dtype} | max_len={self.max_len}")

    def _pool_hidden(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == "cls":
            return last_hidden_state[:, 0]
        elif self.pooling_strategy == "mean":
            return (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        elif self.pooling_strategy == "max":
            return last_hidden_state.masked_fill(~attention_mask.bool().unsqueeze(-1), float("-inf")).max(dim=1).values
        elif self.pooling_strategy == "last":
            idx = attention_mask.sum(dim=1) - 1
            return last_hidden_state[torch.arange(last_hidden_state.size(0)), idx]
        raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    @torch.inference_mode()
    def _encode_chunk(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        ids, mask = (input_ids, attention_mask) if self._is_sharded else (
            input_ids.to(self._input_device, non_blocking=True), attention_mask.to(self._input_device, non_blocking=True))
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True, return_dict=True)
        return self._pool_hidden(out.hidden_states[-1], mask)

    @staticmethod
    def _to_list_f32(t: torch.Tensor) -> List[float]:
        return t.detach().to(torch.float32).cpu().tolist()

    def __call__(self, texts: List[str], batch_size: int = 8, chunk_overlap: int = 128) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        all_embeddings: List[List[float]] = []
        step = max(self.max_len - chunk_overlap, 1)

        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=False, return_tensors="pt", add_special_tokens=True)
            for row in range(enc["input_ids"].size(0)):
                ids, mask = enc["input_ids"][row], enc["attention_mask"][row]
                seq_len = int(mask.sum().item())

                if seq_len <= self.max_len:
                    all_embeddings.append(self._to_list_f32(self._encode_chunk(ids.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)))
                    continue

                pos, chunk_embeds, chunk_weights = 0, [], []
                while pos < seq_len:
                    end = min(pos + self.max_len, seq_len)
                    pooled = self._encode_chunk(ids[pos:end].unsqueeze(0), torch.ones_like(ids[pos:end].unsqueeze(0))).squeeze(0)
                    chunk_embeds.append(pooled.cpu())
                    chunk_weights.append(end - pos)
                    if end == seq_len:
                        break
                    pos += step

                weights = torch.tensor(chunk_weights, dtype=chunk_embeds[0].dtype)
                emb = (torch.stack(chunk_embeds) * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
                all_embeddings.append(self._to_list_f32(emb))
        return all_embeddings

class Llama3ModelEncoder(HFEncoderBase):
    def __init__(self, model_name: str, pooling_strategy: str = "mean", device: Optional[str] = None):
        super().__init__(model_name, pooling_strategy=pooling_strategy, device=device, torch_dtype=torch.bfloat16)

class SentenceTransformerEncoder:
    """Encoder using sentence-transformers library for optimized embedding models."""
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

        self.device = device or ('cuda' if torch.cuda.is_available() else
                                 ('mps' if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else 'cpu'))
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name
        print(f"[SentenceTransformerEncoder] {model_name} loaded on {self.device}")

    def __call__(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=False,
            normalize_embeddings=True
        )

        # Convert to a list of lists
        if isinstance(embeddings, torch.Tensor):
            return embeddings.cpu().tolist()
        return list(embeddings)


class HashingEncoder:
    """Word 1-3 grams (or character n-grams) hashed into `n_features` dimensions, L2-normalized.

    No download and no GPU. A lexical encoder in the spirit of the paper's bag-of-words and n-gram
    reference methods: strong when states are written in a fixed template, and a quick way to
    try the pipeline before encoding with a large language model. On the toy ferry world, word
    n-grams beat character n-grams by a wide margin at Hit@1.
    """

    def __init__(self, n_features: int = 1024, analyzer: str = "word", ngram_range=(1, 3)):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.n_features, self.analyzer, self.ngram_range = n_features, analyzer, tuple(ngram_range)
        self._vec = HashingVectorizer(analyzer=analyzer, ngram_range=self.ngram_range, n_features=n_features,
                                      alternate_sign=False, norm="l2", lowercase=True)

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return self._vec.transform(list(texts)).toarray().astype(np.float32)


def _as_array_encoder(fn: Callable) -> Callable[[Sequence[str]], np.ndarray]:
    def encode(texts: Sequence[str]) -> np.ndarray:
        out = fn(list(texts))
        if isinstance(out, torch.Tensor):
            out = out.detach().to(torch.float32).cpu().numpy()
        out = np.asarray(out, dtype=np.float32)
        if out.ndim != 2 or out.shape[0] != len(texts):
            raise ValueError(f"the encoder returned shape {out.shape} for {len(texts)} texts; "
                             "expected (n_texts, dim)")
        return out
    return encode


def get_encoder(spec: Union[str, Callable], device: Optional[str] = None) -> Callable[[Sequence[str]], np.ndarray]:
    """One callable from a name or a function: texts in, float32 array (n_texts, dim) out.

    "hashing"                        word n-gram hashing (no download)
    a sentence-transformers name     e.g. "BAAI/bge-m3", "sentence-transformers/all-mpnet-base-v2"
    a Llama or Qwen model name       mean-pooled last hidden layer, e.g. "Qwen/Qwen2.5-7B-Instruct"
    any other Hugging Face name      tried with sentence-transformers first
    a callable                       your own encoder, e.g. a lookup into precomputed embeddings
    """
    if callable(spec):
        return _as_array_encoder(spec)
    if not isinstance(spec, str):
        raise TypeError(f"encoder must be a model name or a callable, got {type(spec).__name__}")
    name = spec.lower()
    if name == "hashing":
        return HashingEncoder()
    if "llama" in name or "qwen" in name:
        return _as_array_encoder(Llama3ModelEncoder(spec, device=device))
    return _as_array_encoder(SentenceTransformerEncoder(spec, device=device))

