<!-- =========================================================
     Instacart Retail Project — README (HTML + inline CSS)
     No external icons, no link-buttons
     Copy-paste this entire file into README.md
     ========================================================= -->

<div align="center" style="
  padding:24px 18px;
  border-radius:18px;
  background:linear-gradient(135deg,#06182c 0%, #0b3b7a 45%, #06182c 100%);
  box-shadow:0 14px 40px rgba(0,0,0,.30);
  border:1px solid rgba(255,255,255,.14);
  overflow:hidden;
  position:relative;
">

  <!-- soft animated highlights -->
  <div style="
    position:absolute; inset:-120px -120px auto auto;
    width:280px; height:280px;
    background:radial-gradient(circle at 30% 30%, rgba(120,200,255,.35), rgba(120,200,255,0) 62%);
    filter:blur(6px);
    animation:floatGlow 6.5s ease-in-out infinite;
    pointer-events:none;
  "></div>

  <div style="
    position:absolute; inset:auto auto -140px -140px;
    width:340px; height:340px;
    background:radial-gradient(circle at 60% 60%, rgba(80,255,210,.18), rgba(80,255,210,0) 64%);
    filter:blur(8px);
    animation:floatGlow2 8s ease-in-out infinite;
    pointer-events:none;
  "></div>

  <!-- animated "retail conveyor" line -->
  <div style="
    margin:0 auto 14px auto;
    width:min(980px, 92%);
    height:8px;
    border-radius:999px;
    background:rgba(255,255,255,.10);
    border:1px solid rgba(255,255,255,.12);
    overflow:hidden;
  ">
    <div style="
      height:100%;
      width:45%;
      border-radius:999px;
      background:linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,.55), rgba(255,255,255,0));
      animation:scan 2.2s linear infinite;
    "></div>
  </div>

  <h1 style="margin:0;color:#ffffff;letter-spacing:.2px;font-weight:850;">
    Instacart Retail Analytics
  </h1>

  <p style="margin:10px auto 0;color:rgba(255,255,255,.86);max-width:980px;line-height:1.55;">
    End-to-end retail analytics on the Instacart Market Basket dataset: reproducible ingestion → star schema modeling → transaction baskets →
    association rule mining (Apriori, FP-Growth, Eclat) and utility-aware mining (UP-Tree with a controlled pricing layer).
  </p>

  <!-- animated "basket tokens" -->
  <div style="margin-top:16px;display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">
    <div style="padding:8px 12px;border-radius:999px;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.14);color:#fff;font-weight:650;">
      Bronze
      <span style="display:inline-block;margin-left:8px;width:8px;height:8px;border-radius:50%;background:rgba(255,255,255,.55);animation:pulse 1.8s ease-in-out infinite;"></span>
    </div>
    <div style="padding:8px 12px;border-radius:999px;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.14);color:#fff;font-weight:650;">
      Silver
      <span style="display:inline-block;margin-left:8px;width:8px;height:8px;border-radius:50%;background:rgba(255,255,255,.55);animation:pulse 1.8s ease-in-out infinite 0.35s;"></span>
    </div>
    <div style="padding:8px 12px;border-radius:999px;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.14);color:#fff;font-weight:650;">
      Gold
      <span style="display:inline-block;margin-left:8px;width:8px;height:8px;border-radius:50%;background:rgba(255,255,255,.55);animation:pulse 1.8s ease-in-out infinite 0.7s;"></span>
    </div>
    <div style="padding:8px 12px;border-radius:999px;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.14);color:#fff;font-weight:650;">
      ARM + Utility
      <span style="display:inline-block;margin-left:8px;width:8px;height:8px;border-radius:50%;background:rgba(255,255,255,.55);animation:pulse 1.8s ease-in-out infinite 1.05s;"></span>
    </div>
  </div>

  <div style="
    margin-top:16px;
    padding:12px 14px;
    border-radius:14px;
    background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.12);
    max-width:980px;
    text-align:left;
  ">
    <div style="color:#ffffff;font-weight:750;margin-bottom:6px;">Academic objective (A25)</div>
    <div style="color:rgba(255,255,255,.85);line-height:1.55;">
      Discover product affinity patterns and translate them into business actions (bundling, cross-sell, promo strategy).
      Deliver reproducible notebooks, shareable outputs for dashboards/reporting, and a clear viva narrative.
    </div>
  </div>

  <!-- inline CSS animations -->
  <style>
    @keyframes scan { 
      0% { transform: translateX(-120%); opacity: .35; }
      25% { opacity: .75; }
      50% { opacity: .95; }
      100% { transform: translateX(260%); opacity: .35; }
    }
    @keyframes pulse {
      0%,100% { transform: scale(1); opacity:.45; }
      50% { transform: scale(1.55); opacity:.9; }
    }
    @keyframes floatGlow {
      0%,100% { transform: translate(0,0); opacity:.85; }
      50% { transform: translate(-18px, 14px); opacity:1; }
    }
    @keyframes floatGlow2 {
      0%,100% { transform: translate(0,0); opacity:.75; }
      50% { transform: translate(16px, -12px); opacity:.95; }
    }
  </style>
</div>

<br/>

<div style="
  padding:14px 14px;
  border-radius:16px;
  background:rgba(15,76,129,.10);
  border:1px solid rgba(15,76,129,.22);
">
  <b>Included:</b> notebooks, code, lightweight outputs (CSV), docs, app scaffolding (if present).<br/>
  <b>Not included:</b> Instacart raw dataset + large generated artifacts (keeps the repo clean and reproducible).
</div>

---

## Repository Structure
```text
instacart-retail-project/
├─ notebooks/                 # Main notebooks (run order below)
├─ outputs/                   # Shareable CSV outputs (dashboard-ready)
├─ dashboard/                 # Optional dashboard/app assets
├─ data/                      # Local only (raw/processed) — not committed
├─ docs/                      # Report + documentation
├─ requirements.txt
└─ README.md
