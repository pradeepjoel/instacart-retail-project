<!-- =========================================================
     Instacart Retail Project — README (HTML + inline CSS)
     Copy-paste this entire file into README.md
     ========================================================= -->

<div align="center" style="
  padding:22px 18px;
  border-radius:18px;
  background:linear-gradient(135deg,#071a2d 0%, #0b3b7a 45%, #071a2d 100%);
  box-shadow:0 12px 34px rgba(0,0,0,.28);
  border:1px solid rgba(255,255,255,.14);
">
  <h1 style="margin:0;color:#ffffff;letter-spacing:.2px;font-weight:800;">
    Instacart Retail Analytics
  </h1>

  <p style="margin:10px auto 0;color:rgba(255,255,255,.86);max-width:980px;line-height:1.45;">
    Retail analytics project using the Instacart Market Basket dataset to extract actionable product affinity patterns.
    Includes association rule mining (Apriori, FP-Growth, Eclat), utility-aware mining (UP-Tree with a controlled pricing layer),
    and optional application/report layers for business communication.
  </p>

  <div style="margin-top:12px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.x-1f6feb?style=for-the-badge" />
    <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-f28b00?style=for-the-badge" />
    <img alt="Pandas" src="https://img.shields.io/badge/Pandas-DataFrames-0b1f3a?style=for-the-badge" />
    <img alt="Association Rules" src="https://img.shields.io/badge/ARM-Apriori%20%7C%20FP--Growth%20%7C%20Eclat-0f4c81?style=for-the-badge" />
    <img alt="Utility Mining" src="https://img.shields.io/badge/Utility-UP--Tree-0b1f3a?style=for-the-badge" />
  </div>

  <div style="
    margin-top:14px;
    padding:12px 14px;
    border-radius:14px;
    background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.12);
    max-width:980px;
    text-align:left;
  ">
    <div style="color:#ffffff;font-weight:700;margin-bottom:6px;">Academic objective (A25)</div>
    <div style="color:rgba(255,255,255,.85);line-height:1.5;">
      Translate discovered patterns into business value (bundling, cross-sell, promo strategy).
      Provide reproducible notebooks + shareable outputs for dashboards/reporting and a clear narrative for viva.
    </div>
  </div>
</div>

<br/>

<div style="
  padding:14px 14px;
  border-radius:16px;
  background:rgba(15,76,129,.10);
  border:1px solid rgba(15,76,129,.22);
">
  <b>What is included in GitHub:</b> code, notebooks, lightweight outputs (CSVs), docs, app scaffolding (if present).<br/>
  <b>What is NOT included:</b> the Instacart raw dataset and large generated artifacts (to keep the repo clean and lightweight).
</div>

---

## Contents
- Project deliverables
- Repository structure
- Setup
- Raw data (NOT included)
- Notebook execution order
- Outputs to share with analysts (dashboards)
- Pricing + utility (UP-Tree) note
- Team work distribution

---

## Project Deliverables (Rubric Alignment)
- Reproducible notebooks implementing the full workflow
- Web app / dashboard layer (optional; for demo and stakeholder communication)
- PDF report (project write-up)
- GitHub repository (this repo)

---

## Repository Structure (High Level)
```text
instacart-retail-project/
├─ notebooks/                 # Main work (run order below)
├─ dashboard/                 # Dashboard/web app assets (if used)
├─ outputs/                   # Shareable CSV outputs (lightweight)
├─ data/                      # Local only (raw/processed) — NOT committed
├─ docs/                      # Report + documentation
├─ requirements.txt
└─ README.md
