"""splitting.py -- paragraph-grouped 5-fold stratified CV with inner val split.

Uses ``StratifiedGroupKFold`` keyed on the context-paragraph hash to prevent
data leakage when the same SQuAD paragraph appears in multiple QA pairs.
Inside each fold a ~15% group-disjoint validation slice is carved out (with
a plain-stratified fallback when the grouped slice is degenerate).

``split_data`` returns a list of ``(idx_train, idx_val, idx_test)`` tuples.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

_N_SPLITS = 5
_VAL_FRACTION = 0.15
_SEED = 42


def _paragraph_key(prompt: str) -> str:
    """Hash the context paragraph from a ChatML prompt.

    The user turn contains boilerplate, a context paragraph, and finally the
    question after "Here is the question:". Cutting at that marker yields a
    string that is identical for every question sharing the same paragraph.
    """
    s = str(prompt)
    if "<|im_start|>user" in s:
        s = s.split("<|im_start|>user", 1)[1]
    for marker in ("Here is the question", "<|im_end|>"):
        if marker in s:
            s = s.split(marker, 1)[0]
            break
    return hashlib.md5(s.strip().encode("utf-8")).hexdigest()


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Return a list of ``(idx_train, idx_val, idx_test)`` tuples -- one per fold.

    Uses paragraph-grouped k-fold when the DataFrame has a ``prompt`` column;
    falls back to a plain stratified single split otherwise.
    """
    y = np.asarray(y)

    # Fallback: no DataFrame or no prompt column -- plain stratified split
    if df is None or "prompt" not in df.columns:
        idx = np.arange(len(y))
        idx_tv, idx_te = train_test_split(
            idx,
            test_size=test_size,
            random_state=_SEED,
            stratify=y,
        )
        rel = val_size / (1.0 - test_size)
        idx_tr, idx_va = train_test_split(
            idx_tv,
            test_size=rel,
            random_state=_SEED,
            stratify=y[idx_tv],
        )
        return [(idx_tr, idx_va, idx_te)]

    # Primary path: paragraph-grouped StratifiedGroupKFold
    groups = df["prompt"].astype(str).map(_paragraph_key).to_numpy()
    sgkf = StratifiedGroupKFold(
        n_splits=_N_SPLITS,
        shuffle=True,
        random_state=_SEED,
    )

    folds: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for tr, te in sgkf.split(np.zeros(len(y)), y, groups):
        # Carve a group-disjoint ~15% validation slice from training groups
        # so paragraphs don't leak from train into val either.
        tr_groups = np.array(sorted(set(groups[tr])))
        rng = np.random.default_rng(_SEED + len(folds))
        rng.shuffle(tr_groups)
        n_val_groups = max(1, int(round(_VAL_FRACTION * len(tr_groups))))
        val_groups = set(tr_groups[:n_val_groups].tolist())
        is_val = np.array([g in val_groups for g in groups[tr]])
        idx_val = tr[is_val]
        idx_train = tr[~is_val]

        # Fall back to plain stratified slice if grouped slice is degenerate
        if (
            idx_val.size == 0
            or np.unique(y[idx_val]).size < 2
            or np.unique(y[idx_train]).size < 2
        ):
            idx_train, idx_val = train_test_split(
                tr,
                test_size=_VAL_FRACTION,
                random_state=_SEED,
                stratify=y[tr],
            )

        folds.append((idx_train, idx_val, te))
    return folds
