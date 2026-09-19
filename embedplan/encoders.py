import time
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import litellm
from litellm import completion



class HFEncoderBase:
    def __init__(self, model_name: str, pooling_strategy: str = "last", device: Optional[str] = None,
                 torch_dtype: torch.dtype = torch.bfloat16, attn_impl: Optional[str] = None):
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.device = device or ('cuda' if torch.cuda.is_available() else
                                 ('mps' if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else 'cpu'))

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

        # Convert to list of listsCRRC
        if isinstance(embeddings, torch.Tensor):
            return embeddings.cpu().tolist()
        return list(embeddings)

class ClaudeGenerator:
    def __init__(self, model_name="vertex_ai/claude-3-opus", project=None, location=None):
        self.model_name = model_name
        self.fail_counter = 0
        if project:
            litellm.vertex_project = project
        if location:
            litellm.vertex_location = location

    def __call__(self, batch_prompts: List[str], **kwargs) -> List[str]:
        all_generated_texts = []
        for prompt in batch_prompts:
            while True:
                try:
                    response = completion(model=self.model_name, messages=[{"role": "user", "content": prompt}], **kwargs)
                    all_generated_texts.append(response["choices"][0]["message"]["content"])
                    self.fail_counter = 0
                    break
                except Exception as e:
                    print(f"Completion failed (attempt {self.fail_counter}): {e}")
                    self.fail_counter += 1
                    wait_time = 2 ** self.fail_counter
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        return all_generated_texts
