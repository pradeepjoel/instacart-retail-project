# Retail Analytics & Pattern Mining Dashboard (Single-file, no src imports)
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Retail Analytics & Pattern Mining", layout="wide")

# ---------------------------
# Paths
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------
# Helpers
# ---------------------------
def parse_id_list(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [int(x) for x in str(s).split(",") if x.strip().isdigit()]

def ids_to_names(ids, id_to_name):
    if not ids:
        return ""
    return " + ".join([id_to_name.get(i, f"id_{i}") for i in ids])

@st.cache_data(show_spinner=False)
def load_data():
    dim_products = pd.read_parquet(PROCESSED_DIR / "dim_products.parquet")
    fact = pd.read_parquet(PROCESSED_DIR / "fact_order_items_slim.parquet")
    rules = pd.read_parquet(OUTPUT_DIR / "rules_fpgrowth.parquet")
    prices = pd.read_parquet(PROCESSED_DIR / "product_prices.parquet")
    return dim_products, fact, rules, prices

@st.cache_data(show_spinner=False)
def compute_top_products(fact_df, dim_products_df, top_n: int):
    return (
        fact_df["product_id"]
        .value_counts()
        .head(int(top_n))
        .rename_axis("product_id")
        .reset_index(name="order_item_count")
        .merge(
            dim_products_df[["product_id", "product_name", "department"]],
            on="product_id",
            how="left",
        )
    )

@st.cache_data(show_spinner=False)
def compute_dept_counts(fact_df, dim_products_df, top_n: int):
    return (
        fact_df.merge(dim_products_df[["product_id", "department"]], on="product_id", how="left")
        .groupby("department", as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .head(int(top_n))
    )

@st.cache_data(show_spinner=False)
def compute_filtered_rules(rules_df, min_support: float, min_conf: float, min_lift: float):
    filtered = rules_df[
        (rules_df["support"] >= min_support)
        & (rules_df["confidence"] >= min_conf)
        & (rules_df["lift"] >= min_lift)
    ].copy()
    return filtered.sort_values(["lift", "confidence", "support"], ascending=False)

@st.cache_data(show_spinner=False)
def compute_order_features(fact_df, prices_df):
    f = fact_df[["order_id", "product_id"]]
    p = prices_df[["product_id", "price"]]
    return (
        f.merge(p, on="product_id", how="left")
        .groupby("order_id", as_index=False)
        .agg(
            basket_size=("product_id", "count"),
            total_spent=("price", "sum"),
        )
    )

# ---------------------------
# Load
# ---------------------------
dim_products_df, fact_df, rules_df, prices_df = load_data()

# Reduce RAM (important on Streamlit Cloud)
fact_df = fact_df[["order_id", "product_id", "reordered"]].copy()
fact_df["order_id"] = fact_df["order_id"].astype("int32", copy=False)
fact_df["product_id"] = fact_df["product_id"].astype("int32", copy=False)
fact_df["reordered"] = fact_df["reordered"].astype("int8", copy=False)

prices_df = prices_df[["product_id", "price"]].copy()
prices_df["product_id"] = prices_df["product_id"].astype("int32", copy=False)
prices_df["price"] = prices_df["price"].astype("float32", copy=False)

# Validate required columns
req_fact = {"order_id", "product_id", "reordered"}
req_dim = {"product_id", "product_name", "department"}
req_rules = {"antecedents_str", "consequents_str", "support", "confidence", "lift"}
req_prices = {"product_id", "price"}

missing = (req_fact - set(fact_df.columns)) | (req_dim - set(dim_products_df.columns)) | \
          (req_rules - set(rules_df.columns)) | (req_prices - set(prices_df.columns))

if missing:
    st.error(f"Missing required columns: {sorted(missing)}")
    st.stop()

id_to_name = dict(zip(dim_products_df["product_id"], dim_products_df["product_name"]))

# ---------------------------
# UI
# ---------------------------
st.title("Retail Analytics & Pattern Mining Dashboard")

st.sidebar.header("Filters")
min_support = st.sidebar.slider("Min Support", 0.0, float(max(0.01, rules_df["support"].max())), 0.002, 0.0005)
min_conf = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.30, 0.05)
lift_hi = float(max(2.0, rules_df["lift"].quantile(0.99)))
min_lift = st.sidebar.slider("Min Lift", 1.0, lift_hi, 1.5, 0.5)

top_n_products = st.sidebar.selectbox("Top N products", [10, 20, 30, 50], index=0)
top_n_depts = st.sidebar.selectbox("Top N departments", [10, 15, 20], index=1)

st.sidebar.divider()
perf_mode = st.sidebar.toggle("Performance mode (Cloud stable)", value=True)
st.sidebar.caption("Segmentation runs only when you click the button (prevents crashes).")

tab_overview, tab_rules, tab_viz, tab_seg = st.tabs(
    ["Retail Overview", "Rules Explorer", "Rule Visuals", "Segmentation & Revenue Impact"]
)

# ==========================================================
# Tab 1: Retail Overview
# ==========================================================
with tab_overview:
    unique_orders = int(fact_df["order_id"].nunique())
    unique_products = int(fact_df["product_id"].nunique())
    reorder_rate = float(fact_df["reordered"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Order-Item Rows", f"{len(fact_df):,}")
    c2.metric("Unique Orders", f"{unique_orders:,}")
    c3.metric("Unique Products", f"{unique_products:,}")
    c4.metric("Reorder Rate", f"{reorder_rate:.3f}")

    st.subheader("Top Products by Volume")
    top_products = compute_top_products(fact_df, dim_products_df, int(top_n_products))
    st.dataframe(top_products, use_container_width=True)

    chart_prod = (
        alt.Chart(top_products)
        .mark_bar()
        .encode(
            x=alt.X("order_item_count:Q", title="Order-item count"),
            y=alt.Y("product_name:N", sort="-x", title="Product"),
            tooltip=["product_name", "department", "order_item_count"],
        )
        .properties(height=380)
    )
    st.altair_chart(chart_prod, use_container_width=True)

    st.subheader("Top Departments by Volume")
    dept_counts = compute_dept_counts(fact_df, dim_products_df, int(top_n_depts))

    chart_dept = (
        alt.Chart(dept_counts)
        .mark_bar()
        .encode(
            x=alt.X("size:Q", title="Order-item count"),
            y=alt.Y("department:N", sort="-x", title="Department"),
            tooltip=["department", "size"],
        )
        .properties(height=360)
    )
    st.altair_chart(chart_dept, use_container_width=True)

# ==========================================================
# Tab 2: Rules Explorer
# ==========================================================
with tab_rules:
    st.subheader("Association Rules (FP-Growth)")

    filtered = compute_filtered_rules(rules_df, float(min_support), float(min_conf), float(min_lift))
    st.write(f"Filtered rules: {len(filtered):,}")

    show_k = st.number_input("Show top K rules", min_value=10, max_value=500, value=50, step=10)

    show_df = filtered.head(int(show_k)).copy()
    show_df["antecedent_names"] = show_df["antecedents_str"].apply(lambda x: ids_to_names(parse_id_list(x), id_to_name))
    show_df["consequent_names"] = show_df["consequents_str"].apply(lambda x: ids_to_names(parse_id_list(x), id_to_name))

    view_cols = ["antecedent_names", "consequent_names", "support", "confidence", "lift"]
    st.dataframe(show_df[view_cols], use_container_width=True)

    csv = show_df[view_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download shown rules (CSV)", data=csv, file_name="filtered_rules_topk.csv", mime="text/csv")

# ==========================================================
# Tab 3: Rule Visuals
# ==========================================================
with tab_viz:
    st.subheader("Rule Scatter Plot")

    filtered_v = compute_filtered_rules(rules_df, float(min_support), float(min_conf), float(min_lift))
    if filtered_v.empty:
        st.info("No rules match the current thresholds. Reduce Min Support/Confidence/Lift.")
    else:
        max_points = 2000
        filtered_v = filtered_v.head(max_points).copy()

        filtered_v["A_name"] = filtered_v["antecedents_str"].apply(lambda x: ids_to_names(parse_id_list(x), id_to_name))
        filtered_v["B_name"] = filtered_v["consequents_str"].apply(lambda x: ids_to_names(parse_id_list(x), id_to_name))
        filtered_v["rule"] = filtered_v["A_name"] + "  →  " + filtered_v["B_name"]

        scatter = (
            alt.Chart(filtered_v)
            .mark_circle(size=80)
            .encode(
                x=alt.X("support:Q", title="Support"),
                y=alt.Y("lift:Q", title="Lift"),
                color=alt.Color("confidence:Q", title="Confidence"),
                tooltip=["rule", "support", "confidence", "lift"],
            )
            .properties(height=420)
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

# ==========================================================
# Tab 4: Segmentation & Revenue Impact
# ==========================================================
with tab_seg:
    st.subheader("Revenue Impact (Top Rules Simulation)")
    st.caption("Business idea: promote A → B (recommendations / bundles) and increase confidence to estimate incremental revenue.")

    rules_sel = rules_df[["support", "confidence", "lift", "antecedents_str", "consequents_str"]].copy()
    rules_sel = rules_sel.sort_values(["lift", "confidence", "support"], ascending=False).head(500)

    def rule_label(row):
        a_ids = parse_id_list(row["antecedents_str"])
        b_ids = parse_id_list(row["consequents_str"])
        a_name = ids_to_names(a_ids, id_to_name)
        b_name = ids_to_names(b_ids, id_to_name)
        return f"{a_name} → {b_name} | lift={row['lift']:.2f}, conf={row['confidence']:.2f}, supp={row['support']:.4f}"

    rules_sel["label"] = rules_sel.apply(rule_label, axis=1)

    selected_label = st.selectbox("Select a rule (top 500)", rules_sel["label"].tolist())
    uplift_pct = st.slider("Uplift on confidence (%)", 0, 50, 10, 5, key="rev_uplift_slider")

    chosen = rules_sel.loc[rules_sel["label"] == selected_label].iloc[0]

    N = int(fact_df["order_id"].nunique())
    support = float(chosen["support"])
    confidence = float(chosen["confidence"])

    support_A = support / confidence if confidence > 0 else 0.0
    orders_A = support_A * N

    uplift = uplift_pct / 100.0
    new_conf = confidence * (1.0 + uplift)
    additional_orders = max(0.0, (new_conf - confidence) * orders_A)

    b_ids = parse_id_list(chosen["consequents_str"])
    b_id = int(b_ids[0]) if b_ids else None

    avg_price = 0.0
    if b_id is not None:
        row = prices_df[prices_df["product_id"] == b_id]
        if not row.empty:
            avg_price = float(row["price"].iloc[0])

    additional_revenue = additional_orders * avg_price

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Orders (N)", f"{N:,}")
    k2.metric("Estimated orders containing A", f"{int(orders_A):,}")
    k3.metric("Additional purchases of B", f"{int(additional_orders):,}")
    k4.metric("Estimated incremental revenue (€)", f"{additional_revenue:,.2f}")

    st.caption("Note: support(A) estimated from support(A∩B)/confidence. Approximation is fine for business interpretation.")

    st.divider()
    st.subheader("Customer Segmentation (Order-level)")

    run_seg = True
    if perf_mode:
        run_seg = st.button("Run segmentation (compute now)", type="primary")

    if not run_seg:
        st.info("Segmentation paused for stability. Click the button to compute.")
    else:
        order_features = compute_order_features(fact_df, prices_df)

        order_features["basket_segment"] = pd.cut(
            order_features["basket_size"],
            bins=[0, 5, 15, 50, 200],
            labels=["Small", "Medium", "Large", "Bulk"],
        )

        order_features["spend_segment"] = pd.qcut(
            order_features["total_spent"],
            q=3,
            labels=["Budget", "Standard", "Premium"],
        )

        c1, c2 = st.columns(2)

        with c1:
            st.caption("Basket Segment Distribution")
            basket_counts = (
                order_features["basket_segment"]
                .value_counts()
                .rename_axis("segment")
                .reset_index(name="count")
            )
            chart_seg1 = (
                alt.Chart(basket_counts)
                .mark_bar()
                .encode(
                    x=alt.X("segment:N", title="Basket segment"),
                    y=alt.Y("count:Q", title="Orders"),
                    tooltip=["segment", "count"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart_seg1, use_container_width=True)

        with c2:
            st.caption("Spend Segment Distribution")
            spend_counts = (
                order_features["spend_segment"]
                .value_counts()
                .rename_axis("segment")
                .reset_index(name="count")
            )
            chart_seg2 = (
                alt.Chart(spend_counts)
                .mark_bar()
                .encode(
                    x=alt.X("segment:N", title="Spend segment"),
                    y=alt.Y("count:Q", title="Orders"),
                    tooltip=["segment", "count"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart_seg2, use_container_width=True)
