"""aggregation.py -- feature-block registry for the ensemble probe.

The full feature vector is the concatenation of every entry in ``BLOCKS``.
Each block carries ``(name, extractor, dim)``. Sub-probes in ``probe.py``
reference blocks by name and read contiguous slices via ``BLOCK_SLICES[name]``.

To add a block: write an extractor with signature
``f(hidden_states, attention_mask, response_mask) -> 1-D Tensor`` and append
a ``FeatureBlock(name, extract, dim)`` to ``BLOCKS``. Slices update
automatically.
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from model import MAX_LENGTH, _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Block registry types
# ---------------------------------------------------------------------------


@dataclass
class FeatureBlock:
    name: str
    extract: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    dim: int


# ---------------------------------------------------------------------------
# Module-level state (set lazily; read by extractors via response_mask())
# ---------------------------------------------------------------------------

_RESPONSE_STARTS: list[int] | None = None
_INPUT_IDS_LIST: list[torch.Tensor] | None = None
_REGEN_CACHE: np.ndarray | None = None
_CALL_IDX: int = 0

# Per-call values, refreshed by response_mask() exactly once per sample.
_CURRENT_INPUT_IDS: torch.Tensor | None = None
_CURRENT_REGEN: np.ndarray | None = None

_LM_HEAD_W: torch.Tensor | None = None
_FINAL_NORM_W: torch.Tensor | None = None
_RMSNORM_EPS: float = 1e-6


def _ensure_tokenizations() -> None:
    """Pre-tokenise every (prompt + response) row once. Stores the
    response-start position AND the full input_ids tensor -- the latter is
    required by the actual-logprob block to gather logprobs of the actual
    response tokens at inference.
    """
    global _RESPONSE_STARTS, _INPUT_IDS_LIST
    if _RESPONSE_STARTS is not None:
        return
    tok = AutoTokenizer.from_pretrained(_DEFAULT_MODEL)
    starts: list[int] = []
    ids_list: list[torch.Tensor] = []
    for path in ("./data/dataset.csv", "./data/test.csv"):
        df = pd.read_csv(path)
        for prompt, response in zip(df["prompt"], df["response"]):
            n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
            full_ids = tok(
                prompt + response,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_LENGTH,
            )["input_ids"]
            starts.append(min(n_prompt, MAX_LENGTH - 1))
            ids_list.append(torch.tensor(full_ids, dtype=torch.long))
    _RESPONSE_STARTS = starts
    _INPUT_IDS_LIST = ids_list


# ---------------------------------------------------------------------------
# Regeneration-overlap cache: greedy-regen features (6-d) per dataset row.
# ---------------------------------------------------------------------------


def _pick_regen_batch_size() -> int:
    if not torch.cuda.is_available():
        return 1
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb > 60:
        return 128
    if vram_gb > 30:
        return 128
    if vram_gb > 12:
        return 256
    return 1


class _Top1LogprobCapture(LogitsProcessor):
    """Captures the per-step top-1 logprob on CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.logprobs: list[torch.Tensor] = []

    def __call__(self, input_ids, scores):  # type: ignore[override]
        log_probs = torch.log_softmax(scores.float(), dim=-1)
        top1 = scores.argmax(dim=-1, keepdim=True)
        top1_lp = log_probs.gather(1, top1).squeeze(-1)
        self.logprobs.append(top1_lp.detach().cpu())
        return scores


def _compute_regen_stats_from_ids(
    gen_ids: list[int],
    actual_ids_list: list[int],
    avg_logprob: float,
    first_n: int = 10,
) -> np.ndarray:
    """Build the 6-d regen feature row."""
    if not gen_ids:
        return np.zeros(6, dtype=np.float32)
    actual_ids_set = set(actual_ids_list)
    actual_len = len(actual_ids_list)
    gen_ids_set = set(gen_ids)
    inter = actual_ids_set & gen_ids_set
    union = actual_ids_set | gen_ids_set
    jaccard = len(inter) / max(len(union), 1)
    overlap_ratio = len(inter) / max(actual_len, 1)
    n = min(first_n, actual_len, len(gen_ids))
    if n > 0:
        first_n_match = sum(1 for i in range(n) if gen_ids[i] == actual_ids_list[i]) / n
    else:
        first_n_match = 0.0
    length_ratio = len(gen_ids) / max(actual_len, 1)
    length_diff = float(len(gen_ids) - actual_len)
    return np.array(
        [
            jaccard,
            overlap_ratio,
            first_n_match,
            length_ratio,
            length_diff,
            avg_logprob,
        ],
        dtype=np.float32,
    )


def _run_regen_chunk(
    model,
    tok,
    device,
    sample_indices: list[int],
    all_prompt_ids: list[list[int]],
    all_actual_ids: list[list[int]],
    feats: list,
    max_gen_tokens: int = 256,
    first_n: int = 10,
) -> None:
    """Greedily regenerate a chunk of samples in one batched generate()."""
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id

    valid = [i for i in sample_indices if all_prompt_ids[i]]
    for i in sample_indices:
        if not all_prompt_ids[i]:
            feats[i] = np.zeros(6, dtype=np.float32)
    if not valid:
        return

    max_prompt_len = max(len(all_prompt_ids[i]) for i in valid)
    input_ids_list = [
        [pad_id] * (max_prompt_len - len(all_prompt_ids[i])) + all_prompt_ids[i]
        for i in valid
    ]
    attn_list = [
        [0] * (max_prompt_len - len(all_prompt_ids[i])) + [1] * len(all_prompt_ids[i])
        for i in valid
    ]
    ids_t = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    attn_t = torch.tensor(attn_list, dtype=torch.long, device=device)

    capture = _Top1LogprobCapture()
    with torch.no_grad():
        out = model.generate(
            input_ids=ids_t,
            attention_mask=attn_t,
            max_new_tokens=max_gen_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
            return_dict_in_generate=True,
            output_scores=False,
            logits_processor=[capture],
        )

    new_tokens = out.sequences[:, ids_t.shape[1] :].cpu()
    n_steps = new_tokens.shape[1]
    if capture.logprobs:
        per_step_lp = torch.stack(capture.logprobs, dim=0)
    else:
        per_step_lp = torch.zeros((n_steps, len(valid)))

    for k, i in enumerate(valid):
        seq_k = new_tokens[k].tolist()
        if eos_id is not None and eos_id in seq_k:
            end = seq_k.index(eos_id) + 1
            gen_ids = seq_k[:end]
        else:
            gen_ids = seq_k
        n_real = len(gen_ids)
        if n_real > 0 and per_step_lp.numel() > 0:
            avg_lp = float(per_step_lp[:n_real, k].mean().item())
        else:
            avg_lp = 0.0
        feats[i] = _compute_regen_stats_from_ids(
            gen_ids,
            all_actual_ids[i],
            avg_lp,
            first_n,
        )

    del out, new_tokens, per_step_lp, ids_t, attn_t, capture


def _build_regen_cache_inline() -> None:
    """Build the regen feature cache by greedily regenerating Qwen's
    continuation for every row, then persist to ``regenerations.npz``.
    """
    global _REGEN_CACHE
    if _REGEN_CACHE is not None:
        return

    print(
        "[aggregation] Building regen cache inline (batched greedy generation).",
        flush=True,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        vram_gb = 0.0
        gpu_name = "MPS"
    else:
        device = torch.device("cpu")
        vram_gb = 0.0
        gpu_name = "CPU"
        print(
            "[aggregation] WARNING: no GPU available -- generation will be "
            "extremely slow (hours).",
            file=sys.stderr,
            flush=True,
        )

    batch_size = _pick_regen_batch_size()
    print(
        f"[aggregation]   device={device} ({gpu_name}, "
        f"{vram_gb:.1f} GB) -> batch_size={batch_size}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(_DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        _DEFAULT_MODEL,
        dtype=torch.bfloat16,
    )
    model.eval().to(device)

    print("[aggregation]   pre-tokenising dataset.csv + test.csv ...", flush=True)
    all_prompt_ids: list[list[int]] = []
    all_actual_ids: list[list[int]] = []
    n_train_rows: int | None = None
    for path in ("./data/dataset.csv", "./data/test.csv"):
        if not Path(path).exists():
            print(f"[aggregation]   {path} not found, skipping", flush=True)
            continue
        df = pd.read_csv(path)
        for prompt, response in zip(df["prompt"], df["response"]):
            pids = tok(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=480,
            )["input_ids"]
            rids = tok(response, add_special_tokens=False)["input_ids"]
            all_prompt_ids.append(pids)
            all_actual_ids.append(rids)
        if path.endswith("dataset.csv"):
            n_train_rows = len(all_prompt_ids)

    n_total = len(all_prompt_ids)
    feats: list = [None] * n_total

    order = sorted(range(n_total), key=lambda i: len(all_prompt_ids[i]))

    t0 = time.time()
    chunk_idx = 0
    while chunk_idx * batch_size < n_total:
        chunk = order[chunk_idx * batch_size : (chunk_idx + 1) * batch_size]
        try:
            _run_regen_chunk(
                model,
                tok,
                device,
                chunk,
                all_prompt_ids,
                all_actual_ids,
                feats,
            )
        except torch.cuda.OutOfMemoryError:
            print(
                f"[aggregation]   OOM at batch_size={len(chunk)} (chunk "
                f"{chunk_idx + 1}); falling back to per-sample for this chunk",
                flush=True,
            )
            torch.cuda.empty_cache()
            for i in chunk:
                try:
                    _run_regen_chunk(
                        model,
                        tok,
                        device,
                        [i],
                        all_prompt_ids,
                        all_actual_ids,
                        feats,
                    )
                except Exception as e:
                    print(
                        f"[aggregation]   sample {i}: {type(e).__name__}: {e}, zeros",
                        flush=True,
                    )
                    feats[i] = np.zeros(6, dtype=np.float32)
        except Exception as e:
            print(
                f"[aggregation]   chunk {chunk_idx + 1} error: "
                f"{type(e).__name__}: {e}, zeros for {len(chunk)} samples",
                flush=True,
            )
            for i in chunk:
                feats[i] = np.zeros(6, dtype=np.float32)

        chunk_idx += 1
        n_done = min(chunk_idx * batch_size, n_total)
        report_every = max(1, n_total // 20)
        if (n_done % report_every < batch_size) or n_done == n_total:
            elapsed = time.time() - t0
            eta = (elapsed / max(n_done, 1)) * (n_total - n_done)
            print(
                f"[aggregation]   regen progress: {n_done}/{n_total} "
                f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                flush=True,
            )

    _REGEN_CACHE = np.stack(
        [f if f is not None else np.zeros(6, dtype=np.float32) for f in feats],
        axis=0,
    )

    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(
        f"[aggregation] Regen cache built: shape={_REGEN_CACHE.shape}, "
        f"took {time.time() - t0:.1f}s",
        flush=True,
    )

    if n_train_rows is not None:
        out_path = Path("regenerations.npz")
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_path,
                train=_REGEN_CACHE[:n_train_rows],
                test=_REGEN_CACHE[n_train_rows:],
            )
            print(
                f"[aggregation]   saved cache to {out_path} "
                f"(train={n_train_rows}, test={n_total - n_train_rows})",
                flush=True,
            )
        except Exception as e:
            print(
                f"[aggregation]   could not save cache to {out_path}: {e!r}",
                flush=True,
            )


def _ensure_regen_cache() -> None:
    """Populate the regen feature cache. Resolution order:
    1. Already in memory -> no-op.
    2. ``regenerations.npz`` on disk -> load.
    3. Build inline (also persists to ``regenerations.npz``).
    """
    global _REGEN_CACHE
    if _REGEN_CACHE is not None:
        return
    cache_path = Path("regenerations.npz")
    if cache_path.exists():
        try:
            z = np.load(cache_path)
            _REGEN_CACHE = np.concatenate(
                [z["train"], z["test"]],
                axis=0,
            ).astype(np.float32)
            print(
                f"[aggregation] Loaded regen cache from {cache_path} "
                f"(shape {_REGEN_CACHE.shape})",
                flush=True,
            )
            return
        except Exception as e:
            print(
                f"[aggregation] Failed to load {cache_path}: {e!r}, "
                "falling back to inline build",
                flush=True,
            )
    _build_regen_cache_inline()


# ---------------------------------------------------------------------------
# Per-sample mask + state refresh
# ---------------------------------------------------------------------------


def response_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the boolean mask over response tokens AND refresh per-sample
    globals. Advances the per-sample counter exactly once per call.
    """
    global _CALL_IDX, _CURRENT_INPUT_IDS, _CURRENT_REGEN
    _ensure_tokenizations()
    _ensure_regen_cache()
    assert _RESPONSE_STARTS is not None and _INPUT_IDS_LIST is not None
    assert _REGEN_CACHE is not None
    start = _RESPONSE_STARTS[_CALL_IDX]
    _CURRENT_INPUT_IDS = _INPUT_IDS_LIST[_CALL_IDX]
    _CURRENT_REGEN = _REGEN_CACHE[_CALL_IDX]
    _CALL_IDX += 1
    real = attention_mask.bool()
    pos = torch.arange(real.numel())
    return real & (pos >= start)


def _ensure_lm_head(device: torch.device) -> None:
    """Lazily load Qwen's final RMSNorm + lm_head weights onto ``device``."""
    global _LM_HEAD_W, _FINAL_NORM_W
    if _LM_HEAD_W is not None and _LM_HEAD_W.device == device:
        return
    if _LM_HEAD_W is None:
        m = AutoModelForCausalLM.from_pretrained(
            _DEFAULT_MODEL,
            dtype=torch.bfloat16,
        )
        _LM_HEAD_W = m.lm_head.weight.detach().clone()
        _FINAL_NORM_W = m.model.norm.weight.detach().clone()
        del m
    _LM_HEAD_W = _LM_HEAD_W.to(device)
    _FINAL_NORM_W = _FINAL_NORM_W.to(device)


# ---------------------------------------------------------------------------
# Block extractor factories
# ---------------------------------------------------------------------------


def _resp_mean_at_layer(L: int) -> Callable:
    def extract(hidden_states, attention_mask, rmask):
        layer = hidden_states[L]
        if rmask.any():
            return layer[rmask.to(layer.device)].mean(dim=0)
        last = int(attention_mask.nonzero(as_tuple=False)[-1].item())
        return layer[last]

    return extract


def _last_token_at_layer(L: int) -> Callable:
    def extract(hidden_states, attention_mask, rmask):
        last = int(attention_mask.nonzero(as_tuple=False)[-1].item())
        return hidden_states[L][last]

    return extract


def _logit_lens(layers: tuple[int, ...]) -> Callable:
    """For each layer, project response hidden states through Qwen's final
    RMSNorm + lm_head; extract entropy / top-1 prob / top1-top2 margin
    aggregated as [mean, min, max] over the response span.
    """

    @torch.no_grad()
    def extract(hidden_states, attention_mask, rmask):
        device = hidden_states.device
        _ensure_lm_head(device)
        assert _LM_HEAD_W is not None and _FINAL_NORM_W is not None
        if rmask.any():
            idx = rmask.to(device).nonzero(as_tuple=True)[0]
        else:
            last = int(attention_mask.nonzero(as_tuple=False)[-1].item())
            idx = torch.tensor([last], device=device)
        w = _LM_HEAD_W.float()
        norm_w = _FINAL_NORM_W.float()
        feats: list[torch.Tensor] = []
        for L in layers:
            x = hidden_states[L].index_select(0, idx).float()
            rms = torch.rsqrt((x * x).mean(-1, keepdim=True) + _RMSNORM_EPS)
            normed = x * rms * norm_w
            logits = normed @ w.t()
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            top2 = probs.topk(2, dim=-1).values
            top1 = top2[:, 0]
            margin = top2[:, 0] - top2[:, 1]
            entropy = -(probs * log_probs).sum(-1)
            for vec in (entropy, top1, margin):
                feats.extend([vec.mean(), vec.amin(), vec.amax()])
        return torch.stack(feats, dim=0)

    return extract


def _length_features() -> Callable:
    """``[prompt_tok, response_tok, total_tok, is_truncated]`` (4 scalars)."""

    def extract(hidden_states, attention_mask, rmask):
        device = hidden_states.device
        seq_len = float(attention_mask.sum().item())
        resp_len = float(rmask.sum().item())
        prompt_len = max(seq_len - resp_len, 0.0)
        is_trunc = float(seq_len >= MAX_LENGTH - 0.5)
        return torch.tensor(
            [prompt_len, resp_len, seq_len, is_trunc],
            device=device,
        )

    return extract


def _cross_layer_mean_factory(
    layers: tuple[int, ...] = (12, 13, 14, 15, 16),
    last_k: int = 3,
) -> Callable:
    """Mid-layer pool: per layer, take [last_token, mean(last K real tokens)];
    average across them into a single (hidden_dim,) feature.
    """

    def extract(hidden_states, attention_mask, rmask):
        real_pos = attention_mask.nonzero(as_tuple=False).flatten()
        if real_pos.numel() == 0:
            return hidden_states[0][0] * 0
        last_pos = int(real_pos[-1].item())
        device = hidden_states.device
        real_pos_d = real_pos.to(device)
        n_real = real_pos.numel()
        feats: list[torch.Tensor] = []
        for L in layers:
            layer = hidden_states[L]
            feats.append(layer[last_pos])
            if n_real >= last_k:
                tail = layer.index_select(0, real_pos_d[-last_k:])
            else:
                tail = layer.index_select(0, real_pos_d)
            feats.append(tail.mean(dim=0))
        stacked = torch.stack(feats, dim=0)
        return stacked.mean(dim=0)

    return extract


def _geo_features_factory() -> Callable:
    """101 hand-crafted geometric features."""

    def extract(hidden_states, attention_mask, rmask):
        device = hidden_states.device
        n_layers = hidden_states.size(0)
        last = int(attention_mask.nonzero(as_tuple=False)[-1].item())

        last_tokens = hidden_states[:, last, :]
        norms = last_tokens.norm(dim=-1)

        has_resp = bool(rmask.any())
        rmask_d = rmask.to(device) if has_resp else None
        if has_resp:
            resp = hidden_states[:, rmask_d, :]
            resp_mag = resp.abs().mean(dim=(1, 2))
            resp_mean = resp.mean(dim=1)
        else:
            resp_mag = torch.zeros(n_layers, device=device)
            resp_mean = last_tokens

        drift_last = F.cosine_similarity(last_tokens[:-1], last_tokens[1:], dim=-1)
        drift_mean = F.cosine_similarity(resp_mean[:-1], resp_mean[1:], dim=-1)

        if has_resp:
            last_layer_resp = hidden_states[-1, rmask_d, :]
            if last_layer_resp.size(0) >= 2:
                tok_drift = F.cosine_similarity(
                    last_layer_resp[:-1],
                    last_layer_resp[1:],
                    dim=-1,
                )
                spread = tok_drift.std(unbiased=False).unsqueeze(0)
            else:
                spread = torch.zeros(1, device=device)
        else:
            spread = torch.zeros(1, device=device)

        seq_len = torch.tensor(
            [float(attention_mask.sum().item())],
            device=device,
        )
        resp_len = torch.tensor(
            [float(rmask.sum().item())],
            device=device,
        )

        return torch.cat(
            [norms, resp_mag, drift_last, drift_mean, spread, seq_len, resp_len],
            dim=0,
        )

    return extract


def _actual_logprob_features_factory() -> Callable:
    """Layer-24 logit-lens (9) + actual-token features (6) = 15 scalars."""

    @torch.no_grad()
    def extract(hidden_states, attention_mask, rmask):
        device = hidden_states.device
        _ensure_lm_head(device)
        assert _LM_HEAD_W is not None and _FINAL_NORM_W is not None
        assert _CURRENT_INPUT_IDS is not None, (
            "response_mask must be called before this block"
        )

        # 9 within-layer features at layer 24.
        if rmask.any():
            idx = rmask.to(device).nonzero(as_tuple=True)[0]
        else:
            last = int(attention_mask.nonzero(as_tuple=False)[-1].item())
            idx = torch.tensor([last], device=device)
        h_resp = hidden_states[-1].index_select(0, idx)
        x = h_resp.float()
        rms = torch.rsqrt((x * x).mean(-1, keepdim=True) + _RMSNORM_EPS)
        normed = x * rms * _FINAL_NORM_W.float()
        logits = normed @ _LM_HEAD_W.float().t()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        top2 = probs.topk(2, dim=-1).values
        top1 = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(probs * log_probs).sum(-1)
        within_9 = torch.stack(
            [
                entropy.mean(),
                entropy.amin(),
                entropy.amax(),
                top1.mean(),
                top1.amin(),
                top1.amax(),
                margin.mean(),
                margin.amin(),
                margin.amax(),
            ]
        )

        # 6 actual-token features at layer 24.
        rmask_idx = rmask.nonzero(as_tuple=True)[0]
        rmask_idx = rmask_idx[rmask_idx > 0]
        n_real = int(attention_mask.sum().item())
        rmask_idx = rmask_idx[rmask_idx < n_real]
        if rmask_idx.numel() == 0:
            actual_6 = torch.zeros(6, device=device)
        else:
            pred_idx = (rmask_idx - 1).to(device)
            rmask_idx_d = rmask_idx.to(device)
            h_pred = hidden_states[-1].index_select(0, pred_idx)
            xp = h_pred.float()
            rmsp = torch.rsqrt((xp * xp).mean(-1, keepdim=True) + _RMSNORM_EPS)
            normp = xp * rmsp * _FINAL_NORM_W.float()
            logits_p = normp @ _LM_HEAD_W.float().t()
            log_probs_p = torch.log_softmax(logits_p, dim=-1)
            probs_p = log_probs_p.exp()
            actual_ids = (
                _CURRENT_INPUT_IDS.to(device).index_select(0, rmask_idx_d).unsqueeze(-1)
            )
            actual_lp = log_probs_p.gather(1, actual_ids).squeeze(-1)
            actual_prob = actual_lp.exp()
            top1_prob_p = probs_p.amax(dim=-1)
            top1_tok_p = probs_p.argmax(dim=-1)
            is_argmax = (top1_tok_p == actual_ids.squeeze(-1)).float()
            gap = top1_prob_p - actual_prob
            actual_6 = torch.stack(
                [
                    actual_lp.mean(),
                    actual_lp.amin(),
                    actual_lp.amax(),
                    gap.mean(),
                    gap.amax(),
                    is_argmax.mean(),
                ]
            )

        return torch.cat([within_9, actual_6], dim=0)

    return extract


def _regen_features_factory() -> Callable:
    """Read the per-sample 6-d regen-overlap row from the cache."""

    def extract(hidden_states, attention_mask, rmask):
        device = hidden_states.device
        assert _CURRENT_REGEN is not None, (
            "response_mask must be called before this block"
        )
        return torch.tensor(_CURRENT_REGEN, dtype=torch.float32, device=device)

    return extract


def _stat_features_factory(
    layers: tuple[int, ...] = (12, 16, 20, 24),
) -> Callable:
    """Per-layer distribution-shape statistics of per-response-token L2 norms.
    Output dim = 7 * len(layers).
    """

    def extract(hidden_states, attention_mask, rmask):
        device = hidden_states.device
        n_stats = 7
        n_features = len(layers) * n_stats

        if not bool(rmask.any()):
            return torch.zeros(n_features, device=device)
        rmask_d = rmask.to(device)
        idx = rmask_d.nonzero(as_tuple=True)[0]

        feats: list[torch.Tensor] = []
        for L in layers:
            resp = hidden_states[L].index_select(0, idx).float()
            norms = resp.norm(dim=-1)
            n = norms.numel()

            mu = norms.mean()
            std = norms.std(unbiased=False) if n > 1 else torch.zeros((), device=device)

            if n > 0:
                qs = torch.quantile(
                    norms,
                    torch.tensor([0.25, 0.5, 0.75], device=device),
                )
                p25, p50, p75 = qs[0], qs[1], qs[2]
            else:
                p25 = p50 = p75 = torch.zeros((), device=device)

            if n > 1:
                centered = norms - mu
                m2 = (centered * centered).mean()
                m4 = (centered * centered * centered * centered).mean()
                kurt = m4 / (m2 * m2 + 1e-12) - 3.0
            else:
                kurt = torch.zeros((), device=device)

            if n > 0:
                lp = torch.log_softmax(norms, dim=0)
                p = lp.exp()
                ent = -(p * lp).sum()
            else:
                ent = torch.zeros((), device=device)

            feats.extend([mu, std, p25, p50, p75, kurt, ent])

        return torch.stack(feats, dim=0)

    return extract


def _layer_drift_l2_factory() -> Callable:
    """L2 magnitude of layer-to-layer change in hidden states. Total 48 scalars."""

    def extract(hidden_states, attention_mask, rmask):
        device = hidden_states.device
        last = int(attention_mask.nonzero(as_tuple=False)[-1].item())

        last_tokens = hidden_states[:, last, :].float()
        drift_last = (last_tokens[1:] - last_tokens[:-1]).norm(dim=-1)

        if bool(rmask.any()):
            rmask_d = rmask.to(device)
            resp = hidden_states[:, rmask_d, :].float()
            resp_mean = resp.mean(dim=1)
        else:
            resp_mean = last_tokens
        drift_mean = (resp_mean[1:] - resp_mean[:-1]).norm(dim=-1)

        return torch.cat([drift_last, drift_mean], dim=0)

    return extract


# ---------------------------------------------------------------------------
# Block registry -- edit this to add/remove blocks.
# ---------------------------------------------------------------------------

BLOCKS: list[FeatureBlock] = [
    FeatureBlock("resp_mean_l23", _resp_mean_at_layer(23), 896),
    FeatureBlock("last_tok_l20", _last_token_at_layer(20), 896),
    FeatureBlock("cross_layer_mean", _cross_layer_mean_factory(), 896),
    FeatureBlock("geo_features", _geo_features_factory(), 101),
    FeatureBlock("logit_lens_24", _logit_lens((24,)), 9),
    FeatureBlock("length", _length_features(), 4),
    FeatureBlock("actual_logprob", _actual_logprob_features_factory(), 15),
    FeatureBlock("regen_features", _regen_features_factory(), 6),
    FeatureBlock("stat_features", _stat_features_factory(), 28),
    FeatureBlock("layer_drift_l2", _layer_drift_l2_factory(), 48),
    FeatureBlock("layer14_last_tok", _last_token_at_layer(14), 896),
    FeatureBlock("layer14_resp_mean", _resp_mean_at_layer(14), 896),
]

BLOCK_SLICES: dict[str, slice] = {}
_offset = 0
for _b in BLOCKS:
    BLOCK_SLICES[_b.name] = slice(_offset, _offset + _b.dim)
    _offset += _b.dim
TOTAL_DIM: int = _offset


# ---------------------------------------------------------------------------
# Public surface (called by the frozen solution.py)
# ---------------------------------------------------------------------------


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    rmask = response_mask(attention_mask)
    feats = [b.extract(hidden_states, attention_mask, rmask) for b in BLOCKS]
    return torch.cat(feats, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    return aggregate(hidden_states, attention_mask)


# ---------------------------------------------------------------------------
# Eager regen-cache build at module load.
# ---------------------------------------------------------------------------


def _eager_build_caches_at_module_load() -> None:
    try:
        from probe import SUB_PROBES  # noqa: WPS433
    except Exception as e:
        print(
            f"[aggregation] Could not import probe at module-load to check "
            f"sub-probes: {e!r}. Regen cache will be built lazily on first "
            "aggregate() call (may compete with solution.py's Qwen for GPU memory).",
            flush=True,
        )
        return

    if any("regen_features" in cfg.blocks for cfg in SUB_PROBES):
        _ensure_regen_cache()


_eager_build_caches_at_module_load()
