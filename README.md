<!-- =========================================================
README — Data-Driven Retail Insights for Cost Savings & Revenue Growth
Repo: instacart-retail-project
========================================================== -->

<div align="center">

# 🛒 Data-Driven Retail Insights for Cost Savings & Revenue Growth  
### Instacart Market Basket Analytics • Customer Segmentation • Utility-Aware Pattern Mining • Streamlit Dashboard

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-2ea44f?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Wrangling-1f77b4?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-Association%20Rules-6f42c1" />
  <img src="https://img.shields.io/badge/Dataset-Instacart%20(Kaggle)-0aa0ff" />
  <img src="https://img.shields.io/badge/Status-Active-success" />
</p>

<!-- Optional: add your banner image if you have one -->
<!-- <img src="docs/banner.png" width="900" /> -->

---

### 🎯 Core Business Question  
**“How much money can I save or earn if I listen to these insights?”**  
We answer this with **revenue simulations**, **bundle recommendations**, **customer segmentation**, and **promotion ROI guidance**.

<br/>

<!-- =========================
Animated Flow (pure SVG)
========================= -->
<svg width="920" height="130" viewBox="0 0 920 130" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Animated pipeline">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#00E5FF"/>
      <stop offset="50%" stop-color="#7C4DFF"/>
      <stop offset="100%" stop-color="#00E676"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000" flood-opacity="0.35"/>
    </filter>
    <style>
      .card { fill: #0b1220; stroke: rgba(255,255,255,0.18); stroke-width: 1.2; rx: 18; filter:url(#shadow); }
      .title { fill: #e6edf3; font: 700 14px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; }
      .sub { fill: rgba(230,237,243,0.72); font: 12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; }
      .line { fill: none; stroke: url(#g); stroke-width: 3.5; stroke-linecap: round; stroke-dasharray: 10 10; animation: dash 2.2s linear infinite; }
      .dot { fill: #00E5FF; opacity: 0.9; animation: pulse 1.2s ease-in-out infinite; }
      @keyframes dash { to { stroke-dashoffset: -40; } }
      @keyframes pulse { 0%,100%{ transform: translateY(0); opacity:0.55;} 50%{ transform: translateY(-2px); opacity:1;} }
    </style>
  </defs>

  <!-- Cards -->
  <rect class="card" x="30"  y="35" width="180" height="70" rx="18"/>
  <rect class="card" x="250" y="35" width="190" height="70" rx="18"/>
  <rect class="card" x="480" y="35" width="190" height="70" rx="18"/>
  <rect class="card" x="710" y="35" width="180" height="70" rx="18"/>

  <!-- Titles -->
  <text class="title" x="55"  y="63">INGEST</text>
  <text class="sub"   x="55"  y="84">Kaggle Instacart data</text>

  <text class="title" x="275" y="63">TRANSFORM</text>
  <text class="sub"   x="275" y="84">Bronze → Silver → Gold</text>

  <text class="title" x="505" y="63">MINE & MODEL</text>
  <text class="sub"   x="505" y="84">Apriori • FP-Growth • Eclat • UP-Tree</text>

  <text class="title" x="735" y="63">DELIVER</text>
  <text class="sub"   x="735" y="84">Streamlit dashboard + insights</text>

  <!-- Lines -->
  <path class="line" d="M210 70 C230 70, 230 70, 250 70"/>
  <path class="line" d="M440 70 C460 70, 460 70, 480 70"/>
  <path class="line" d="M670 70 C690 70, 690 70, 710 70"/>

  <!-- Dots -->
  <circle class="dot" cx="230" cy="70" r="4"/>
  <circle class="dot" cx="460" cy="70" r="4"/>
  <circle class="dot" cx="690" cy="70" r="4"/>
</svg>

<br/>

<p>
  <a href="YOUR_STREAMLIT_APP_URL"><b>🚀 Live App</b></a> •
  <a href="YOUR_DEMO_VIDEO_URL"><b>🎥 Demo Video</b></a> •
  <a href="#-quickstart"><b>⚡ Quickstart</b></a> •
  <a href="#-team--work-distribution"><b>👥 Team</b></a>
</p>

</div>

---

## 📌 Overview
This project uses the **Instacart Online Grocery Basket Analysis dataset** (3M+ orders, 200K+ customers) to generate **actionable retail insights** that help shop owners **save money and grow revenue** through smarter bundling, segmentation, and promotion strategies.  

**Instructor:** Assan Sanogo  
**Academic Context:** DSTI (Applied MSc tracks)  

---

## ✅ Objectives (What we deliver)
From the project brief :contentReference[oaicite:1]{index=1}, our tool is designed to help shop owners:

- **Boost sales** via **smart product bundling & upselling**
- **Predict purchases** (next-basket / repeat purchase behavior)
- **Segment customers** for targeted marketing (budget vs premium, frequent vs irregular)
- **Design bundles that maximize revenue**
- **Simulate revenue impact** (money saved/earned if insights are adopted)
- **Measure promotion efficiency** (ROI of targeted vs untargeted discounts)

---

## 🧠 Methods
### 1) Association Rule Mining (Market Basket)
- **Apriori** (classic frequent itemsets)
- **FP-Growth** (scalable frequent pattern mining)
- **Eclat** (depth-first, efficient for certain sparsity patterns)
- **UP-Tree (Utility Pattern Tree)**: goes beyond frequency by incorporating **utility/value**, enabling monetization-focused bundles :contentReference[oaicite:2]{index=2}

### 2) Customer Segmentation
- Basket size & variability
- Shopping frequency (frequent/irregular)
- Behavior clustering features engineered from orders/products

### 3) Revenue / Savings Simulation
- Compare current practices vs recommended bundles/promotions
- Estimate uplift and opportunity cost using basket & pricing logic

---

## 🧱 Data Layers (Bronze → Silver → Gold)
**Bronze (Raw):** source data (orders, products, aisles, departments)  
**Silver (Clean):** joins, quality checks, typed columns, dedup, validated keys  
**Gold (Business-ready):** star schema facts/dims, transactions table, ML features, mining inputs  

> Optional enrichment (as suggested in the brief): integrate real pricing sources (e.g., Open Food Facts Prices) when available :contentReference[oaicite:3]{index=3}.

---

## 🗂️ Repository Structure
```bash
instacart-retail-project/
├─ dashboard/                 # Streamlit app
├─ data/processed/            # curated parquet/csv outputs for app & models
├─ notebooks/                 # end-to-end notebooks
│  ├─ 01_ingestion_validation_new.ipynb
│  ├─ 02_data_modeling_new.ipynb
│  ├─ 03_build_transactions.ipynb
│  ├─ 04_exploratory_data_analysis.ipynb
│  ├─ 05-association_rules.ipynb
│  ├─ 06_business_insights_from_association-rule-mining.ipynb
│  ├─ 07_clients_segmentation.ipynb
│  ├─ feature_extraction_XGBoost.ipynb
│  └─ fp_gowth_apriori.ipynb
├─ docs/                      # report / supporting docs
├─ outputs/                   # charts, tables, intermediate exports
└─ requirements.txt
