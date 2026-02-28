from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path("data") / "processed"

@st.cache_data(show_spinner=False)
def read_rules_csv() -> pd.DataFrame:
    """
    Loads association rules.
    Expected (best case) columns: ['rule','support','confidence','lift'].
    """
    candidates = [
        DATA_DIR / "association_rules_fp_0.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)

    raise FileNotFoundError(
        "Could not find association rules CSV. Expected: data/processed/association_rules_fp_0.csv"
    )

@st.cache_data(show_spinner=False)
def read_product_prices() -> pd.DataFrame | None:
    """
    Optional prices table.
    We will try to use it if present to estimate basket values.
    """
    p = DATA_DIR / "product_prices.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None
