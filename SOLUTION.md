# Hallucination Detection in Small Language Models

## Overview

Binary classification of LLM hallucinations from Qwen2.5-0.5B internal
hidden-state representations. The probe reads multi-layer hidden states,
logit-lens projections, and regeneration-overlap features, then classifies
each response as truthful or hallucinated using a hierarchical meta-average
ensemble of 12 sub-probes.

## Architecture

### Feature extraction (aggregation.py)

12 feature blocks (~3695 dimensions total):

| Block | Dim | Description |
|-------|-----|-------------|
| `resp_mean_l23` | 896 | Response-token mean at layer 23 |
| `last_tok_l20` | 896 | Last real token at layer 20 |
| `cross_layer_mean` | 896 | Cross-layer pool (layers 12-16) |
| `geo_features` | 101 | Per-layer L2 norms, cosine drifts, spread |
| `logit_lens_24` | 9 | Layer-24 entropy / top-1 / margin |
| `length` | 4 | Prompt/response/total token counts + truncation flag |
| `actual_logprob` | 15 | Logit-lens (9) + actual-token logprob features (6) |
| `regen_features` | 6 | Greedy regeneration overlap (Jaccard, match rate, etc.) |
| `stat_features` | 28 | Per-layer norm distribution stats at layers 12/16/20/24 |
| `layer_drift_l2` | 48 | L2 magnitude of layer-to-layer hidden-state changes |
| `layer14_last_tok` | 896 | Last real token at layer 14 |
| `layer14_resp_mean` | 896 | Response-token mean at layer 14 |

**Key design choices:**
- Layer 14 captures the middle-layer truthfulness signal, which is strongest
  in intermediate transformer layers rather than the final layer.
- Response-mask aware pooling isolates the model's answer from the prompt
  context.
- Regeneration overlap provides an orthogonal signal by comparing greedy-
  decoded output against the original response.

### Probe classifier (probe.py)

12 sub-probes spanning 5 classifier families:

| Name | Feature block(s) | Classifier | Scaler |
|------|-------------------|------------|--------|
| `cross_layer_mean_lr` | cross_layer_mean | LR(C=0.01) | Robust+clip |
| `geo_gbt` | last_tok_l20 + geo_features | HistGBT | None |
| `resp_mean_lr` | resp_mean_l23 | LR(C=0.01) | Robust+clip |
| `actual_logprob_gbt` | actual_logprob | HistGBT | None |
| `regen_overlap_gbt` | regen_features | HistGBT | None |
| `multi_c_lr_resp_mean` | resp_mean_l23 | Multi-C LR | Robust+clip |
| `extra_trees_geo` | last_tok_l20 + geo_features | ExtraTrees(400) | None |
| `stat_drift_lr` | stat_features + layer_drift_l2 | LR(C=0.01) | Standard |
| `mlp_pool_mlp` | resp_mean_l23 | MLP(256, early-stop) | Robust+clip |
| `layer14_lr` | layer14_last_tok | LR(C=0.01) | Robust+clip |
| `layer14_resp_mean_lr` | layer14_resp_mean | LR(C=0.01) | Robust+clip |
| `layer14_seed_ens_mlp` | layer14_last_tok | 11-MLP seed ensemble | Robust+clip |

**Combiner:** Hierarchical meta-average -- 10 sub-probes are weight-averaged
(weights derived from OOF accuracy), then simple-mean averaged with 5
standalone sub-probes (3 deliberately duplicated for higher effective weight).

**Preprocessing:** Hidden-state sub-probes use RobustScaler (median/IQR)
with hard clipping at +/-5, which is critical for handling the "massive
activation" dimensions present in Qwen2.5-0.5B intermediate layers.

**Threshold tuning:** Out-of-fold (OOF) on ~551 training samples rather
than on the ~83-sample validation slice. `fit_hyperparameters` is a no-op.

### Cross-validation (splitting.py)

5-fold `StratifiedGroupKFold` keyed on the context-paragraph hash (MD5).
Prevents data leakage when the same SQuAD paragraph appears in multiple QA
pairs. Group-disjoint ~15% inner validation split with plain-stratified
fallback.

## Files modified

| File | Description |
|------|-------------|
| `splitting.py` | Paragraph-grouped StratifiedGroupKFold |
| `aggregation.py` | 12 feature blocks with response-mask awareness |
| `probe.py` | 12-sub-probe hierarchical ensemble with RobustScaler+clip |
