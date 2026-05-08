"""probe.py -- meta-average ensemble of sub-probes over the feature blocks
declared in ``aggregation.py``.

Architecture:
    12 sub-probes spanning 5 classifier families (LR, HistGBT, ExtraTrees,
    Multi-C LR, MLP) operate on different feature subsets. Hidden-state
    sub-probes use RobustScaler + clip(+/-5) to handle massive-activation
    dimensions in intermediate layers. A parallel inner 5-fold OOF pass
    derives (a) the weight vector for the top-portion average and (b) the
    accuracy-best decision threshold.

Public surface (required by the frozen evaluate.py / solution.py):
    fit(X, y), fit_hyperparameters(X_val, y_val), predict(X), predict_proba(X).
``predict_proba`` returns shape ``(n, 2)`` with column 1 = P(hallucinated).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from aggregation import BLOCK_SLICES


RANDOM_STATE: int = 42
OOF_FOLDS: int = 5

# RobustScaler clipping range for massive-activation dimensions
_ROBUST_CLIP: float = 5.0


# ---------------------------------------------------------------------------
# Scaler type enum for sub-probes
# ---------------------------------------------------------------------------

SCALER_NONE = "none"
SCALER_STANDARD = "standard"
SCALER_ROBUST = "robust"  # RobustScaler + clip(+/-5) for massive-activation dims


@dataclass
class SubProbeConfig:
    name: str
    blocks: tuple[str, ...]
    factory: Callable[[], Any]
    scaler_type: str = SCALER_STANDARD  # "none", "standard", or "robust"


def _make_scaler(scaler_type: str, X: np.ndarray) -> Any:
    """Create and fit a scaler, or return None."""
    if scaler_type == SCALER_NONE:
        return None
    if scaler_type == SCALER_ROBUST:
        scaler = RobustScaler()
        scaler.fit(X)
        return scaler
    # default: StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def _apply_scaler(scaler: Any, X: np.ndarray, scaler_type: str) -> np.ndarray:
    """Transform X through scaler, optionally clipping for robust mode."""
    if scaler is None:
        return X
    X_out = scaler.transform(X)
    if scaler_type == SCALER_ROBUST:
        X_out = np.clip(X_out, -_ROBUST_CLIP, _ROBUST_CLIP)
    return np.ascontiguousarray(X_out)


# ---------------------------------------------------------------------------
# Multi-C bagged LogisticRegression
# ---------------------------------------------------------------------------

_MULTI_C_VALUES: tuple[float, ...] = (0.001, 0.01, 0.05, 0.1, 0.3, 1.0)


class _MultiCLR:
    """Average LogisticRegression over a fixed grid of C values."""

    def __init__(self, C_values: tuple[float, ...] = _MULTI_C_VALUES, **lr_kwargs):
        self.C_values = tuple(C_values)
        self.lr_kwargs = dict(lr_kwargs)
        self.lr_kwargs.pop("C", None)
        self.clfs: list[LogisticRegression] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MultiCLR":
        self.clfs = [
            LogisticRegression(C=C, **self.lr_kwargs).fit(X, y) for C in self.C_values
        ]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ps = np.stack([c.predict_proba(X)[:, 1] for c in self.clfs])
        p = ps.mean(axis=0)
        return np.stack([1.0 - p, p], axis=1)


def _multi_c_lr_factory(**lr_kwargs) -> Callable[[], _MultiCLR]:
    return lambda: _MultiCLR(C_values=_MULTI_C_VALUES, **lr_kwargs)


# ---------------------------------------------------------------------------
# Small two-layer MLP probe (early-stopped on an inner 15% split)
# ---------------------------------------------------------------------------


class _MLPProbe:
    """Linear(D, hidden) -> ReLU -> Dropout -> Linear(hidden, 1), trained with
    AdamW + BCEWithLogitsLoss on an inner 85/15 split with early stopping.
    """

    def __init__(
        self,
        hidden: int = 256,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 64,
        max_epochs: int = 300,
        patience: int = 20,
        random_state: int = 42,
    ) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self._net: nn.Sequential | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MLPProbe":
        X_tv, X_h, y_tv, y_h = train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=self.random_state,
            stratify=y,
        )
        X_tv_t = torch.from_numpy(X_tv).float()
        y_tv_t = torch.from_numpy(y_tv.astype(np.float32))
        X_h_t = torch.from_numpy(X_h).float()
        y_h_t = torch.from_numpy(y_h.astype(np.float32))

        torch.manual_seed(self.random_state)
        self._net = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )

        n_pos = int(y_tv.sum())
        n_neg = len(y_tv) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        rng = np.random.default_rng(self.random_state)
        n_train = X_tv_t.size(0)

        best_val_loss = float("inf")
        best_state: dict | None = None
        epochs_no_improve = 0

        for _ in range(self.max_epochs):
            self._net.train()
            perm = rng.permutation(n_train)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X_tv_t[idx]
                yb = y_tv_t[idx]
                optimizer.zero_grad()
                logits = self._net(xb).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self._net.eval()
            with torch.no_grad():
                val_loss = criterion(self._net(X_h_t).squeeze(-1), y_h_t).item()
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self._net.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None, "MLP not fitted yet."
        X_t = torch.from_numpy(X).float()
        with torch.no_grad():
            prob_pos = torch.sigmoid(self._net(X_t).squeeze(-1)).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)


def _mlp_probe_factory(**kwargs) -> Callable[[], _MLPProbe]:
    return lambda: _MLPProbe(**kwargs)


# ---------------------------------------------------------------------------
# Seed-ensemble MLP
#
# Trains N_NETS identical MLPs from different seeds, averages sigmoid outputs.
# Full-batch training (no mini-batching, no early stopping) works well for
# the 896-d intermediate-layer features.
# ---------------------------------------------------------------------------

_SEED_ENS_N_NETS: int = 11
_SEED_ENS_EPOCHS: int = 200
_SEED_ENS_HIDDEN: int = 256
_SEED_ENS_LR: float = 1e-3


class _SeedEnsembleMLP:
    """Ensemble of N identical MLPs trained from different random seeds.
    Full-batch training with BCEWithLogitsLoss(pos_weight).
    """

    def __init__(
        self,
        n_nets: int = _SEED_ENS_N_NETS,
        hidden: int = _SEED_ENS_HIDDEN,
        epochs: int = _SEED_ENS_EPOCHS,
        lr: float = _SEED_ENS_LR,
        base_seed: int = 42,
    ) -> None:
        self.n_nets = n_nets
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.base_seed = base_seed
        self._nets: list[nn.Sequential] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SeedEnsembleMLP":
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self._nets = []
        for i in range(self.n_nets):
            torch.manual_seed(self.base_seed + i)
            net = nn.Sequential(
                nn.Linear(X.shape[1], self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, 1),
            )
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
            net.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                criterion(net(X_t).squeeze(-1), y_t).backward()
                optimizer.step()
            net.eval()
            self._nets.append(net)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._nets, "SeedEnsembleMLP not fitted yet."
        X_t = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            probs = np.mean(
                [torch.sigmoid(net(X_t).squeeze(-1)).numpy() for net in self._nets],
                axis=0,
            )
        prob_pos = np.asarray(probs, dtype=np.float64).reshape(-1)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)


def _seed_ensemble_mlp_factory(**kwargs) -> Callable[[], _SeedEnsembleMLP]:
    return lambda: _SeedEnsembleMLP(**kwargs)


# ---------------------------------------------------------------------------
# Sub-probe registry
# ---------------------------------------------------------------------------

SUB_PROBES: list[SubProbeConfig] = [
    SubProbeConfig(
        name="cross_layer_mean_lr",
        blocks=("cross_layer_mean",),
        factory=lambda: LogisticRegression(
            C=0.01,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        scaler_type=SCALER_ROBUST,
    ),
    SubProbeConfig(
        name="geo_gbt",
        blocks=("last_tok_l20", "geo_features"),
        factory=lambda: HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
        scaler_type=SCALER_NONE,
    ),
    SubProbeConfig(
        name="resp_mean_lr",
        blocks=("resp_mean_l23",),
        factory=lambda: LogisticRegression(
            C=0.01,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        scaler_type=SCALER_ROBUST,
    ),
    SubProbeConfig(
        name="actual_logprob_gbt",
        blocks=("actual_logprob",),
        factory=lambda: HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.04,
            max_depth=4,
            l2_regularization=1.5,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=RANDOM_STATE,
        ),
        scaler_type=SCALER_NONE,
    ),
    SubProbeConfig(
        name="regen_overlap_gbt",
        blocks=("regen_features",),
        factory=lambda: HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.04,
            max_depth=4,
            l2_regularization=1.5,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=RANDOM_STATE,
        ),
        scaler_type=SCALER_NONE,
    ),
    SubProbeConfig(
        name="multi_c_lr_resp_mean",
        blocks=("resp_mean_l23",),
        factory=_multi_c_lr_factory(
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        scaler_type=SCALER_ROBUST,
    ),
    SubProbeConfig(
        name="extra_trees_geo",
        blocks=("last_tok_l20", "geo_features"),
        factory=lambda: ExtraTreesClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        scaler_type=SCALER_NONE,
    ),
    SubProbeConfig(
        name="stat_drift_lr",
        blocks=("stat_features", "layer_drift_l2"),
        factory=lambda: LogisticRegression(
            C=0.01,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        scaler_type=SCALER_STANDARD,
    ),
    SubProbeConfig(
        name="mlp_pool_mlp",
        blocks=("resp_mean_l23",),
        factory=_mlp_probe_factory(),
        scaler_type=SCALER_ROBUST,
    ),
    SubProbeConfig(
        name="layer14_lr",
        blocks=("layer14_last_tok",),
        factory=lambda: LogisticRegression(
            C=0.01,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        scaler_type=SCALER_ROBUST,
    ),
    SubProbeConfig(
        name="layer14_resp_mean_lr",
        blocks=("layer14_resp_mean",),
        factory=lambda: LogisticRegression(
            C=0.01,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        scaler_type=SCALER_ROBUST,
    ),
    SubProbeConfig(
        name="layer14_seed_ens_mlp",
        blocks=("layer14_last_tok",),
        factory=_seed_ensemble_mlp_factory(
            n_nets=11,
            hidden=256,
            epochs=200,
            lr=1e-3,
            base_seed=42,
        ),
        scaler_type=SCALER_ROBUST,
    ),
]


# Sub-probes weight-combined into the top-portion average.
META_TOP_NAMES: tuple[str, ...] = (
    "cross_layer_mean_lr",
    "geo_gbt",
    "resp_mean_lr",
    "actual_logprob_gbt",
    "regen_overlap_gbt",
    "multi_c_lr_resp_mean",
    "extra_trees_geo",
    "stat_drift_lr",
    "layer14_lr",
    "layer14_resp_mean_lr",
)

# Sub-probes that additionally enter the final simple mean as standalones.
# Deliberately duplicating the strongest probes gives them higher effective
# weight in the final mean -- this was found optimal by exhaustive subset search.
META_STANDALONE_NAMES: tuple[str, ...] = (
    "cross_layer_mean_lr",
    "geo_gbt",
    "regen_overlap_gbt",
    "mlp_pool_mlp",
    "layer14_seed_ens_mlp",
)


# ---------------------------------------------------------------------------
# Meta-average ensemble probe
# ---------------------------------------------------------------------------


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._sub_states: list[tuple[SubProbeConfig, Any, Any]] = []
        self._top_weights: np.ndarray | None = None
        self._top_idx: list[int] | None = None
        self._standalone_idx: list[int] | None = None
        self._threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "HallucinationProbe is an sklearn-style ensemble; "
            "use predict() / predict_proba() instead of forward()."
        )

    @staticmethod
    def _slice(X: np.ndarray, blocks: tuple[str, ...]) -> np.ndarray:
        if len(blocks) == 1:
            return X[:, BLOCK_SLICES[blocks[0]]]
        return np.concatenate([X[:, BLOCK_SLICES[b]] for b in blocks], axis=1)

    def _sub_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((X.shape[0], len(self._sub_states)), dtype=np.float64)
        for j, (cfg, scaler, clf) in enumerate(self._sub_states):
            X_sub = self._slice(X, cfg.blocks)
            X_pred = _apply_scaler(scaler, X_sub, cfg.scaler_type)
            out[:, j] = clf.predict_proba(X_pred)[:, 1]
        return out

    def _resolve_meta_indices(self) -> tuple[list[int], list[int]]:
        name_to_idx = {cfg.name: i for i, cfg in enumerate(SUB_PROBES)}
        missing = [
            n for n in META_TOP_NAMES + META_STANDALONE_NAMES if n not in name_to_idx
        ]
        if missing:
            raise ValueError(
                f"meta_avg ensemble references sub-probes {missing} that are "
                f"not in SUB_PROBES. Available: {list(name_to_idx)}"
            )
        return (
            [name_to_idx[n] for n in META_TOP_NAMES],
            [name_to_idx[n] for n in META_STANDALONE_NAMES],
        )

    def _combine(self, sp: np.ndarray) -> np.ndarray:
        """Hierarchical mean: weighted top portion + simple mean with standalones."""
        assert self._top_weights is not None
        assert self._top_idx is not None and self._standalone_idx is not None
        top_proba = sp[:, self._top_idx] @ self._top_weights
        stand = sp[:, self._standalone_idx]
        n_components = 1 + stand.shape[1]
        return (top_proba + stand.sum(axis=1)) / n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        # 1. Fit final sub-probes on the full training set.
        self._sub_states = []
        for cfg in SUB_PROBES:
            X_sub = self._slice(X, cfg.blocks)
            scaler = _make_scaler(cfg.scaler_type, X_sub)
            X_fit = _apply_scaler(scaler, X_sub, cfg.scaler_type)
            clf = cfg.factory()
            clf.fit(X_fit, y)
            self._sub_states.append((cfg, scaler, clf))

        # 2. Inner 5-fold OOF pass for threshold tuning + weight derivation.
        sp_oof = self._compute_oof_subproba(X, y)

        # 3. Derive top-portion weights from per-sub-probe OOF accuracy.
        self._top_idx, self._standalone_idx = self._resolve_meta_indices()
        top_oof = sp_oof[:, self._top_idx]
        top_accs = np.array(
            [_best_threshold_acc(top_oof[:, j], y) for j in range(top_oof.shape[1])]
        )
        shifted = np.maximum(top_accs - 0.5, 1e-3)
        self._top_weights = shifted / shifted.sum()

        # 4. Tune the decision threshold on OOF combined probabilities.
        oof_combined = self._combine(sp_oof)
        self._set_threshold_from_probs(oof_combined, np.asarray(y).astype(int))
        return self

    def _compute_oof_subproba(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(y)
        sp_oof = np.zeros((n, len(SUB_PROBES)), dtype=np.float64)
        skf = StratifiedKFold(
            n_splits=OOF_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        for tr, va in skf.split(np.arange(n), y):
            for j, cfg in enumerate(SUB_PROBES):
                X_sub_tr = self._slice(X[tr], cfg.blocks)
                scaler = _make_scaler(cfg.scaler_type, X_sub_tr)
                X_fit = _apply_scaler(scaler, X_sub_tr, cfg.scaler_type)
                clf = cfg.factory()
                clf.fit(X_fit, y[tr])
                X_sub_va = self._slice(X[va], cfg.blocks)
                X_pred = _apply_scaler(scaler, X_sub_va, cfg.scaler_type)
                sp_oof[va, j] = clf.predict_proba(X_pred)[:, 1]
        return sp_oof

    def fit_hyperparameters(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "HallucinationProbe":
        # No-op. The OOF threshold tuned in fit() on ~551 OOF samples is
        # more stable than re-tuning on the ~83-sample val slice.
        return self

    def _set_threshold_from_probs(
        self,
        probs: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Pick the accuracy-best threshold; tie-break toward 0.5."""
        cands = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_t, best_acc = 0.5, -1.0
        for t in cands:
            acc = accuracy_score(y, (probs >= t).astype(int))
            if acc > best_acc or (acc == best_acc and abs(t - 0.5) < abs(best_t - 0.5)):
                best_acc, best_t = acc, float(t)
        self._threshold = best_t

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        sp = self._sub_proba(X)
        p = self._combine(sp)
        return np.stack([1.0 - p, p], axis=1)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _best_threshold_acc(probs: np.ndarray, y: np.ndarray) -> float:
    cands = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best = -1.0
    for t in cands:
        a = accuracy_score(y, (probs >= t).astype(int))
        if a > best:
            best = a
    return best
