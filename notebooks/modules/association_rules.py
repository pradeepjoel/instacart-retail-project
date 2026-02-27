# association_rules.py
# ------------------------------------------------------------
# Reusable utilities for association rule mining on Instacart
# (Temporal split: train mined, test evaluated)
#
# Algorithms:
# - Apriori (mlxtend)
# - FP-Growth (mlxtend)
# - Eclat (pure Python vertical format)
# - UP-Tree via SPMF (true utility mining)
#
# Evaluation:
# - HitRate@K, Precision@K, Recall@K, Coverage@K
#
# Author: (Data Scientist Team)
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import time
import math
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np


# =========================
# General helpers
# =========================

@dataclass
class RunResult:
    frequent_itemsets: Optional[pd.DataFrame]
    rules: Optional[pd.DataFrame]
    meta: Dict[str, Union[int, float, str]]


def timed(fn):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        t1 = time.time()
        if isinstance(out, RunResult):
            out.meta["runtime_sec"] = round(t1 - t0, 4)
            return out
        return out
    return wrapper


# =========================
# Loading & transactions
# =========================

def load_temporal_parquets(
    train_path: Union[str, Path],
    test_path: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load temporal split parquet files containing at least: order_id, product_id."""
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    required = {"order_id", "product_id"}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        raise ValueError(f"Parquets must contain columns {required}. "
                         f"Got train={set(train.columns)}, test={set(test.columns)}")
    return train, test


def build_transactions(
    op_df: pd.DataFrame,
    order_col: str = "order_id",
    item_col: str = "product_id",
    unique_items: bool = True,
) -> pd.Series:
    """
    Return a Series indexed by order_id with list of items (product_ids).
    unique_items=True ensures each product appears once per basket (binary presence).
    """
    if unique_items:
        tx = op_df.groupby(order_col)[item_col].apply(lambda x: list(pd.unique(x)))
    else:
        tx = op_df.groupby(order_col)[item_col].apply(list)
    return tx


# =========================
# One-hot encoding (Apriori / FP-Growth)
# =========================

def transactions_to_onehot(
    transactions: Sequence[Sequence[int]],
    sparse: bool = False,
) -> pd.DataFrame:
    """
    Convert list-of-lists transactions to one-hot DataFrame using mlxtend TransactionEncoder.
    """
    try:
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError as e:
        raise ImportError("mlxtend is required for one-hot encoding. Install with: pip install mlxtend") from e

    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions, sparse=sparse)
    df = pd.DataFrame.sparse.from_spmatrix(arr, columns=te.columns_) if sparse else pd.DataFrame(arr, columns=te.columns_)
    # ensure boolean
    return df.astype(bool)


# =========================
# Apriori
# =========================

@timed
def run_apriori(
    df_onehot: pd.DataFrame,
    min_support: float = 0.005,
    metric: str = "confidence",
    min_threshold: float = 0.3,
    max_len: Optional[int] = None,
) -> RunResult:
    """
    Run Apriori + association rules using mlxtend.
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError as e:
        raise ImportError("mlxtend is required for Apriori. Install with: pip install mlxtend") from e

    itemsets = apriori(
        df_onehot,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
        low_memory=True
    )
    rules = association_rules(itemsets, metric=metric, min_threshold=min_threshold)

    meta = {
        "algorithm": "apriori",
        "min_support": min_support,
        "metric": metric,
        "min_threshold": min_threshold,
        "n_itemsets": int(itemsets.shape[0]),
        "n_rules": int(rules.shape[0]),
    }
    return RunResult(itemsets, rules, meta)


# =========================
# FP-Growth
# =========================

@timed
def run_fpgrowth(
    df_onehot: pd.DataFrame,
    min_support: float = 0.005,
    metric: str = "confidence",
    min_threshold: float = 0.3,
    max_len: Optional[int] = None,
) -> RunResult:
    """
    Run FP-Growth + association rules using mlxtend.
    """
    try:
        from mlxtend.frequent_patterns import fpgrowth, association_rules
    except ImportError as e:
        raise ImportError("mlxtend is required for FP-Growth. Install with: pip install mlxtend") from e

    itemsets = fpgrowth(
        df_onehot,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )
    rules = association_rules(itemsets, metric=metric, min_threshold=min_threshold)

    meta = {
        "algorithm": "fp-growth",
        "min_support": min_support,
        "metric": metric,
        "min_threshold": min_threshold,
        "n_itemsets": int(itemsets.shape[0]),
        "n_rules": int(rules.shape[0]),
    }
    return RunResult(itemsets, rules, meta)


# =========================
# Eclat (vertical format, pure Python)
# =========================

def _build_vertical_db(transactions: pd.Series) -> Tuple[Dict[int, Set[int]], int]:
    """
    Vertical DB: product_id -> set(order_id)
    Returns vertical db + total number of transactions.
    """
    vertical: Dict[int, Set[int]] = {}
    for order_id, items in transactions.items():
        for it in items:
            vertical.setdefault(int(it), set()).add(int(order_id))
    return vertical, int(len(transactions))


def _eclat_recursive(
    items: List[Tuple[int, Set[int]]],
    prefix: Tuple[int, ...],
    min_count: int,
    max_len: Optional[int],
    out: List[Tuple[Tuple[int, ...], int]],
):
    """
    Eclat recursion generating frequent itemsets with tidset intersection.
    items: list of (item, tidset) already filtered by min_count
    """
    for i in range(len(items)):
        item, tidset = items[i]
        new_prefix = prefix + (item,)
        out.append((new_prefix, len(tidset)))

        if max_len is not None and len(new_prefix) >= max_len:
            continue

        suffix = []
        for j in range(i + 1, len(items)):
            item2, tidset2 = items[j]
            inter = tidset & tidset2
            if len(inter) >= min_count:
                suffix.append((item2, inter))

        if suffix:
            _eclat_recursive(suffix, new_prefix, min_count, max_len, out)


@timed
def run_eclat(
    transactions: pd.Series,
    min_support: float = 0.005,
    max_len: Optional[int] = None,
) -> RunResult:
    """
    Run Eclat to get frequent itemsets (no rules by default).
    You can derive rules from itemsets using derive_rules_from_itemsets().
    """
    vertical, n_tx = _build_vertical_db(transactions)
    min_count = max(1, int(math.ceil(min_support * n_tx)))

    # initial frequent 1-itemsets
    items = [(it, tids) for it, tids in vertical.items() if len(tids) >= min_count]
    # sort for deterministic output (by support desc then item id)
    items.sort(key=lambda x: (-len(x[1]), x[0]))

    out: List[Tuple[Tuple[int, ...], int]] = []
    _eclat_recursive(items, prefix=tuple(), min_count=min_count, max_len=max_len, out=out)

    itemsets = pd.DataFrame(
        {
            "itemsets": [frozenset(t[0]) for t in out],
            "support": [t[1] / n_tx for t in out],
            "support_count": [t[1] for t in out],
        }
    ).sort_values(["support", "support_count"], ascending=False).reset_index(drop=True)

    meta = {
        "algorithm": "eclat",
        "min_support": min_support,
        "min_count": min_count,
        "n_transactions": n_tx,
        "n_itemsets": int(itemsets.shape[0]),
        "n_rules": 0,
    }
    return RunResult(frequent_itemsets=itemsets, rules=None, meta=meta)


def derive_rules_from_itemsets(
    itemsets_df: pd.DataFrame,
    metric: str = "confidence",
    min_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Convert frequent itemsets (with 'support' and 'itemsets') into association rules using mlxtend.
    Works for Eclat output or any itemset DataFrame shaped like mlxtend output.
    """
    try:
        from mlxtend.frequent_patterns import association_rules
    except ImportError as e:
        raise ImportError("mlxtend is required to derive rules. Install with: pip install mlxtend") from e

    # mlxtend expects columns: ['support','itemsets']
    df = itemsets_df[["support", "itemsets"]].copy()
    rules = association_rules(df, metric=metric, min_threshold=min_threshold)
    return rules


# =========================
# UP-Tree via SPMF (true utility mining)
# =========================

def build_spmf_utility_file(
    op_with_utility: pd.DataFrame,
    output_path: Union[str, Path],
    order_col: str = "order_id",
    item_col: str = "product_id",
    utility_col: str = "utility",
    item_separator: str = " ",
):
    """
    Build a utility transaction file for SPMF in the common format:
    item1 item2 item3 : TU : u1 u2 u3

    Where:
    - TU is transaction utility (sum of utilities in that transaction)
    - u1,u2,u3 are utilities for each corresponding item
    """
    df = op_with_utility[[order_col, item_col, utility_col]].copy()
    if df[utility_col].isna().any():
        raise ValueError("Utility column contains NaN. Fill or drop before exporting.")

    df[item_col] = df[item_col].astype(int)
    df[utility_col] = df[utility_col].astype(float)

    grouped = df.groupby(order_col)

    lines = []
    for oid, g in grouped:
        items = g[item_col].tolist()
        utils = g[utility_col].tolist()
        tu = float(np.sum(utils))

        items_str = item_separator.join(map(str, items))
        utils_str = item_separator.join(map(lambda x: f"{x:.6f}", utils))
        line = f"{items_str}:{tu:.6f}:{utils_str}"
        lines.append(line)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


@timed
def run_uptree_spmf(
    spmf_jar_path: Union[str, Path],
    input_utility_file: Union[str, Path],
    output_file: Union[str, Path],
    min_utility: Union[int, float],
    item_separator: str = " ",
) -> RunResult:
    """
    Run UP-Tree via SPMF jar.
    This produces utility patterns, not standard confidence/lift rules.
    We return the raw output path in meta; parsing depends on algorithm output format.

    Command format depends on SPMF version. Common call:
      java -jar spmf.jar run UPTree input.txt output.txt minUtility separator

    If your SPMF expects different parameters, adjust here.
    """
    spmf_jar_path = str(spmf_jar_path)
    input_utility_file = str(input_utility_file)
    output_file = str(output_file)

    cmd = [
        "java", "-jar", spmf_jar_path,
        "run", "UPTree",
        input_utility_file,
        output_file,
        str(min_utility),
        item_separator
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    meta = {
        "algorithm": "up-tree (spmf)",
        "min_utility": float(min_utility),
        "spmf_stdout": proc.stdout[-2000:],
        "spmf_stderr": proc.stderr[-2000:],
        "output_file": output_file,
        "returncode": proc.returncode,
    }

    if proc.returncode != 0:
        raise RuntimeError(
            "SPMF UP-Tree failed.\n"
            f"Return code: {proc.returncode}\n"
            f"STDERR:\n{proc.stderr}\n"
            f"STDOUT:\n{proc.stdout}\n"
        )

    # UP-Tree output is patterns, not association_rules DataFrame
    return RunResult(frequent_itemsets=None, rules=None, meta=meta)


# =========================
# Predictive evaluation (Next-basket)
# =========================

def rules_to_recommender(
    rules_df: pd.DataFrame,
    sort_by: str = "lift",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Prepare a rules table sorted for recommendation use.
    rules_df must contain antecedents and consequents as frozensets.
    """
    if rules_df is None or rules_df.empty:
        return rules_df
    if sort_by not in rules_df.columns:
        raise ValueError(f"{sort_by} not in rules columns: {rules_df.columns.tolist()}")
    return rules_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)


def recommend_from_basket(
    basket: Set[int],
    rules_df: pd.DataFrame,
    k: int = 10,
) -> List[int]:
    """
    Recommend up to k items using rule consequents whose antecedents are subset of basket.
    Returns a ranked list of product_ids.
    """
    recs: List[int] = []
    seen = set(basket)

    for _, r in rules_df.iterrows():
        ant: Set[int] = set(r["antecedents"])
        if ant.issubset(basket):
            for c in r["consequents"]:
                c_int = int(c)
                if c_int not in seen:
                    recs.append(c_int)
                    seen.add(c_int)
                    if len(recs) >= k:
                        return recs
    return recs


def evaluate_recommender(
    rules_df: pd.DataFrame,
    test_transactions: pd.Series,
    k: int = 10,
    max_orders: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate on test baskets:
    - HitRate@K: % orders where at least 1 recommended item is in basket
    - Precision@K: avg(#hits / K)
    - Recall@K: avg(#hits / |basket|)
    - Coverage@K: unique recommended items / total unique items in test
    """
    if rules_df is None or rules_df.empty:
        return {"HitRate@K": 0.0, "Precision@K": 0.0, "Recall@K": 0.0, "Coverage@K": 0.0}

    n = 0
    hits_any = 0
    precision_sum = 0.0
    recall_sum = 0.0
    all_recs: Set[int] = set()
    test_items: Set[int] = set()

    it = test_transactions.items()
    if max_orders is not None:
        it = list(it)[:max_orders]

    for _, items in it:
        basket = set(map(int, items))
        test_items |= basket
        recs = recommend_from_basket(basket, rules_df, k=k)

        all_recs |= set(recs)
        hit_items = set(recs) & basket
        if len(hit_items) > 0:
            hits_any += 1

        precision_sum += (len(hit_items) / k) if k > 0 else 0.0
        recall_sum += (len(hit_items) / len(basket)) if len(basket) > 0 else 0.0
        n += 1

    coverage = (len(all_recs) / len(test_items)) if len(test_items) > 0 else 0.0
    return {
        "HitRate@K": hits_any / n if n else 0.0,
        "Precision@K": precision_sum / n if n else 0.0,
        "Recall@K": recall_sum / n if n else 0.0,
        "Coverage@K": coverage,
    }
def evaluate_recommender_proper(
    rules_df,
    test_transactions,
    k=10,
    hide_ratio=0.5,
    random_state=42
):
    import random
    random.seed(random_state)

    hits_any = 0
    precision_sum = 0
    recall_sum = 0
    n = 0

    for _, items in test_transactions.items():
        basket = list(items)

        if len(basket) < 2:
            continue

        # Hide part of basket
        n_hide = max(1, int(len(basket) * hide_ratio))
        hidden = set(random.sample(basket, n_hide))
        observed = set(basket) - hidden

        # Generate recommendations
        recs = recommend_from_basket(observed, rules_df, k=k)

        hit_items = set(recs) & hidden

        if len(hit_items) > 0:
            hits_any += 1

        precision_sum += len(hit_items) / k
        recall_sum += len(hit_items) / len(hidden)
        n += 1

    return {
        "HitRate@K": hits_any / n if n else 0,
        "Precision@K": precision_sum / n if n else 0,
        "Recall@K": recall_sum / n if n else 0,
    }

# =========================
# Business-friendly rule formatting
# =========================

def load_product_lookup(
    products_csv: Union[str, Path],
    columns: Tuple[str, ...] = ("product_id", "product_name", "aisle_id", "department_id"),
) -> pd.DataFrame:
    df = pd.read_csv(products_csv)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in products file: {missing}")
    return df[list(columns)].copy()


def attach_product_names(
    rules_df: pd.DataFrame,
    products_lookup: pd.DataFrame,
    id_col: str = "product_id",
    name_col: str = "product_name",
) -> pd.DataFrame:
    """
    Adds readable string columns for antecedents/consequents using product_name.
    """
    if rules_df is None or rules_df.empty:
        return rules_df

    lookup = products_lookup.set_index(id_col)[name_col].to_dict()

    def names(fs: frozenset) -> str:
        return ", ".join([lookup.get(int(x), str(int(x))) for x in fs])

    out = rules_df.copy()
    out["antecedents_names"] = out["antecedents"].apply(names)
    out["consequents_names"] = out["consequents"].apply(names)
    return out


def top_rules(
    rules_df: pd.DataFrame,
    sort_by: str = "lift",
    n: int = 20,
    min_len_antecedent: int = 1,
    max_len_antecedent: Optional[int] = None,
) -> pd.DataFrame:
    """
    Quick helper to filter and rank rules for insight extraction.
    """
    if rules_df is None or rules_df.empty:
        return rules_df

    df = rules_df.copy()
    df["len_antecedent"] = df["antecedents"].apply(lambda x: len(x))
    df = df[df["len_antecedent"] >= min_len_antecedent]
    if max_len_antecedent is not None:
        df = df[df["len_antecedent"] <= max_len_antecedent]

    if sort_by not in df.columns:
        raise ValueError(f"{sort_by} not in rules columns: {df.columns.tolist()}")

    return df.sort_values(sort_by, ascending=False).head(n).reset_index(drop=True)

def enhance_transaction(
    op_df: pd.DataFrame,
    coverage: float = 0.80,
    order_col: str = "order_id",
    item_col: str = "product_id",
    return_stats: bool = True
):
    """
    Apply Pareto-based filtering to keep only products that together
    account for a specified cumulative purchase coverage.

    Parameters
    ----------
    op_df : DataFrame
        Order-product dataframe (must contain order_id and product_id)
    coverage : float
        Desired cumulative purchase coverage (0 < coverage <= 1)
        Example: 0.80 keeps products explaining 80% of total purchases
    order_col : str
        Name of order column
    item_col : str
        Name of product column
    return_stats : bool
        Whether to return diagnostic statistics

    Returns
    -------
    filtered_df : DataFrame
        Filtered order-product dataframe
    stats : dict (optional)
        Diagnostics about reduction and retained coverage
    """

    if not 0 < coverage <= 1:
        raise ValueError("coverage must be between 0 and 1")

    # Compute product frequency
    product_freq = op_df[item_col].value_counts()

    # Compute cumulative purchase share
    product_share = product_freq / product_freq.sum()
    cumulative_share = product_share.cumsum()

    # Select products within coverage threshold
    selected_products = cumulative_share[cumulative_share <= coverage].index

    # Filter dataset
    filtered_df = op_df[op_df[item_col].isin(selected_products)].copy()

    if not return_stats:
        return filtered_df

    stats = {
        "coverage_target": coverage,
        "original_products": int(op_df[item_col].nunique()),
        "retained_products": int(filtered_df[item_col].nunique()),
        "original_transactions": int(op_df[order_col].nunique()),
        "retained_transactions": int(filtered_df[order_col].nunique()),
        "purchase_volume_retained": round(filtered_df.shape[0] / op_df.shape[0], 4),
        "dimensionality_reduction_ratio":
            round(1 - (filtered_df[item_col].nunique() / op_df[item_col].nunique()), 4)
    }

    return filtered_df, stats

# ============================================================
# Association Rules: Improvements Pack
# 1) Weighted recommender (confidence * lift)
# 2) Lower min_support slightly
# 3) Filter rules with lift > 1
# 4) Keep multi-item antecedents (len >= 2)
# 5) Evaluate again (hide part of basket, predict missing)
# ============================================================

# --- If you already have these from your module, keep using them ---
# from association_rules import (
#     build_transactions, transactions_to_onehot,
#     run_apriori, run_fpgrowth, run_eclat, derive_rules_from_itemsets
# )

# ------------------------------------------------------------
# A) Rule filtering utilities
# ------------------------------------------------------------

def filter_rules_for_prediction(
    rules_df: pd.DataFrame,
    min_lift: float = 1.0,
    min_confidence: float = 0.2,
    min_support: float = 0.0,
    min_antecedent_len: int = 2,
) -> pd.DataFrame:
    """
    Filter rules to improve predictive relevance.
    """
    if rules_df is None or rules_df.empty:
        return rules_df

    df = rules_df.copy()
    df["ante_len"] = df["antecedents"].apply(len)

    keep = (
        (df["lift"] > min_lift) &
        (df["confidence"] >= min_confidence) &
        (df["support"] >= min_support) &
        (df["ante_len"] >= min_antecedent_len)
    )

    return df.loc[keep].sort_values(["confidence", "lift"], ascending=False).reset_index(drop=True)


# ------------------------------------------------------------
# B) Weighted recommender
# ------------------------------------------------------------

def recommend_weighted_from_rules(
    observed_items: set,
    rules_df: pd.DataFrame,
    k: int = 10,
    score_mode: str = "conf_lift",
) -> list:
    """
    Recommend items by aggregating scores from all matching rules.
    score_mode:
      - "conf_lift" = confidence * lift
      - "conf"      = confidence
      - "lift"      = lift
    """
    if rules_df is None or rules_df.empty or not observed_items:
        return []

    scores = {}
    seen = set(observed_items)

    for _, r in rules_df.iterrows():
        ant = set(r["antecedents"])
        if ant.issubset(observed_items):
            if score_mode == "conf":
                s = float(r["confidence"])
            elif score_mode == "lift":
                s = float(r["lift"])
            else:
                s = float(r["confidence"]) * float(r["lift"])

            for c in r["consequents"]:
                c = int(c)
                if c not in seen:
                    scores[c] = scores.get(c, 0.0) + s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in ranked[:k]]


# ------------------------------------------------------------
# C) Evaluation (proper: hide part of basket)
# ------------------------------------------------------------

def evaluate_weighted_recommender(
    rules_df: pd.DataFrame,
    test_transactions: pd.Series,  # order_id -> list(product_id)
    k: int = 10,
    hide_ratio: float = 0.5,
    random_state: int = 42,
    score_mode: str = "conf_lift",
    max_orders: int | None = None,
) -> dict:
    """
    Evaluate recommender by hiding a portion of each test basket.
    - observed = remaining items
    - hidden   = target items
    Compute HitRate@K, Precision@K, Recall@K.
    """
    rng = np.random.default_rng(random_state)

    hits_any = 0
    precision_sum = 0.0
    recall_sum = 0.0
    n = 0

    items_iter = list(test_transactions.items())
    if max_orders is not None:
        items_iter = items_iter[:max_orders]

    for _, items in items_iter:
        basket = list(map(int, items))
        if len(basket) < 2:
            continue

        n_hide = max(1, int(len(basket) * hide_ratio))
        hidden = set(rng.choice(basket, size=n_hide, replace=False))
        observed = set(basket) - hidden

        recs = recommend_weighted_from_rules(observed, rules_df, k=k, score_mode=score_mode)
        hit_items = set(recs) & hidden

        if hit_items:
            hits_any += 1

        precision_sum += len(hit_items) / k
        recall_sum += len(hit_items) / len(hidden)
        n += 1

    return {
        "HitRate@K": hits_any / n if n else 0.0,
        "Precision@K": precision_sum / n if n else 0.0,
        "Recall@K": recall_sum / n if n else 0.0,
        "n_eval_orders": n
    }