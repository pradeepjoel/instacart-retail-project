import pandas as pd
import streamlit as st
import altair as alt


def render_revenue_impact_tab(
    *,
    rules_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    id_to_name: dict[int, str],
    n_orders: int
):
    st.subheader("Revenue Impact (Top Rules Simulation)")

    st.caption(
        "Estimate incremental revenue if we promote rule A → B and increase its confidence."
    )

    uplift_pct = st.slider("Uplift on confidence (%)", 0, 50, 10, 5)

    # Select top rules
    rules_sel = rules_df.sort_values(
        ["lift", "confidence", "support"],
        ascending=False
    ).head(200).copy()

    def parse_ids(s):
        if pd.isna(s) or str(s).strip() == "":
            return []
        return [int(x) for x in str(s).split(",") if x.strip().isdigit()]

    def ids_to_names(ids):
        return " + ".join([id_to_name.get(i, f"product_{i}") for i in ids])

    rules_sel["A_ids"] = rules_sel["antecedents_str"].apply(parse_ids)
    rules_sel["B_ids"] = rules_sel["consequents_str"].apply(parse_ids)

    rules_sel["rule"] = (
        rules_sel["A_ids"].apply(ids_to_names)
        + " → "
        + rules_sel["B_ids"].apply(ids_to_names)
    )

    selected = st.selectbox("Select a rule", rules_sel["rule"])

    chosen = rules_sel[rules_sel["rule"] == selected].iloc[0]

    support = float(chosen["support"])
    confidence = float(chosen["confidence"])

    # Approximate support(A)
    support_A = support / confidence if confidence > 0 else 0.0
    orders_A = support_A * n_orders

    uplift = uplift_pct / 100.0
    new_conf = confidence * (1.0 + uplift)
    additional_orders = (new_conf - confidence) * orders_A

    # Use first consequent product price
    b_ids = chosen["B_ids"]
    b_id = b_ids[0] if b_ids else None

    avg_price = 0.0
    if b_id is not None:
        row = prices_df[prices_df["product_id"] == b_id]
        if not row.empty:
            avg_price = float(row["price"].iloc[0])

    additional_revenue = additional_orders * avg_price

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders (N)", f"{n_orders:,}")
    c2.metric("Orders containing A", f"{int(orders_A):,}")
    c3.metric("Additional purchases of B", f"{int(additional_orders):,}")
    c4.metric("Incremental Revenue (€)", f"{additional_revenue:,.2f}")

    st.caption(
        "Note: support(A) estimated from support(A∩B)/confidence. "
        "For multi-item antecedents this is approximate."
    )
