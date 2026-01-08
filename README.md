# 📊 Data-Driven Retail Insights for Cost Savings & Revenue Growth

This repository contains the full implementation of an applied data analytics project developed as part of the Applied MSc in Data Analytics / Data Science & Artificial Intelligence / Data Engineering & Artificial Intelligence programs at DSTI, under the supervision of Assan Sanogo.

The project leverages large-scale retail transaction data to demonstrate how data science can directly translate into measurable cost savings and revenue growth for shop owners.

🎯 Project Goal

The central business question addressed by this project is:

__“If I adopt these data-driven insights, how much money can I save or earn — and why?”__

Through advanced analytics, machine learning, and business-oriented dashboards, the project delivers actionable recommendations that help retailers:

* Increase sales through smart product bundling and upselling

* Predict customer purchases

* Segment customers to tailor marketing strategies

Design revenue-maximizing bundles and promotions

## 📦 Dataset

The project is based on the Instacart Online Grocery Basket Analysis Dataset, which includes:

Over 3 million orders from 200,000+ customers

Detailed customer behavior (purchase frequency, repeat orders)

Product hierarchy (aisles, departments, categories)

This dataset has been widely used in academic and applied research, making it a robust foundation for both educational exploration and real-world retail optimization.

🧠 Data Science & Analytics Approach

The project explores and compares multiple association rule mining techniques to uncover product relationships:

Apriori – classic frequent itemset mining

Eclat – efficient depth-first search algorithm

FP-Growth – scalable approach for large datasets

UP-Tree (Utility Pattern Tree) – utility-aware mining incorporating product value and profit

While traditional algorithms focus on frequency alone, UP-Tree introduces monetary utility, allowing the team to identify which product combinations truly drive revenue or cost savings. A comparative analysis highlights the added business value of utility-aware mining.

🔧 Data Enrichment

To move from patterns to monetary impact, the dataset may be enriched with real-world product prices (e.g. via Open Food Facts Prices):

Handling missing price information through informed assumptions

Computing the actual financial value of product associations

Producing a clean, structured dataset combining transactions, prices, and product attributes

📈 Business Application & Dashboard

A user-friendly analytical tool/dashboard is designed to support decision-making, featuring:

Customer Insights

Customer segmentation (budget vs premium shoppers)

Buyer types (frequent vs irregular)

Basket size and basket variability

Product & Revenue Insights

Frequent product bundles and co-purchases

Bundle typologies and upselling opportunities

Revenue simulations (potential gains or losses under different strategies)

Promotion efficiency analysis (ROI of targeted vs untargeted discounts)

The dashboard directly answers the shop owner’s core question by quantifying the financial impact of data-driven decisions.

👥 Collaborative & Educational Value

The project reflects a collaborative, end-to-end data workflow:

Data Scientists → association mining, predictive modeling, utility-aware analytics

Data Engineers → data enrichment and integration of external sources

Business Analysts → financial interpretation and impact-driven storytelling

The final deliverable is a simple, business-oriented prototype illustrating how data science can create tangible value in retail.

📁 Repository Deliverables

This repository includes:

Jupyter Notebook(s) / Python scripts covering the full data science pipeline

Complete web application source code

Project report (PDF)

Short demo video of the web application

Well-structured GitHub repository for collaboration and reproducibility

🏆 Evaluation Criteria

The project is evaluated based on:

Machine learning pipeline implementation

Web application completeness

Quality of reporting and presentation

GitHub repository organization

Bonus: best model performance in class
