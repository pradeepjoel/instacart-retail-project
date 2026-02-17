# Instacart Retail Analytics Project

This repository contains the complete **data engineering and analytics pipeline**
for the Instacart Market Basket Analysis project.

Raw and processed datasets are **intentionally not included** in this repository
due to file size constraints and data engineering best practices.

---


---

## How to Clone the Repository

Each team member should clone the repository locally using:

```bash
git clone https://github.com/pradeepjoel/instacart-retail-project.git
cd instacart-retail-project

## Raw Data (Not Included in GitHub)

The raw Instacart dataset is **not stored in GitHub**.

### Download the dataset from
- Kaggle — *Instacart Market Basket Analysis*

After downloading, create the following folder and place all CSV files inside it:

data/raw/


### Expected raw files
- orders.csv
- products.csv
- order_products__prior.csv
- order_products__train.csv
- aisles.csv
- departments.csv

---

## Notebook Execution Order (Mandatory)

Run notebooks **in the following order**.

### 1️⃣ 01_ingestion_validation.ipynb
- Reads raw CSV files from `data/raw/`
- Performs schema checks and basic validation
- Prepares clean base datasets

---

### 2️⃣ 02_data_modeling.ipynb
- Builds dimensional and fact-style tables
- Structures data for analytical use

---

### 3️⃣ 03_build_transactions.ipynb
- Builds transaction-level basket data
- Applies frequency and basket-size filtering
- Prepares datasets optimized for association rule mining

Generated locally:

data/processed/transactions_small_named.parquet


This file is generated locally and is **not committed** to GitHub.

---

### 4️⃣ 04_association_rules.ipynb
- Uses transaction data generated in Notebook 03
- Computes item pairs and association metrics
- Calculates support, confidence, and lift

outputs/association_rules_named.csv


---

## Files to Use by Role

### Data Analysts

Use the following file:



outputs/association_rules_named.csv


Suitable for:
- Power BI / Tableau dashboards
- Pivot tables
- Product affinity analysis
- Business insight reporting

---

### Data Scientists

Use:
- Output from Notebook 04
- Association rule metrics (support, confidence, lift)

Responsibilities include:
- Rule filtering and evaluation
- Business interpretation
- Recommendation logic
- Experimenting with thresholds if required

---

### Data Engineering (Completed)

The data engineering scope includes:
- Data ingestion and validation
- Data modeling
- Performance-aware processing
- Reproducible notebook pipeline
- Clean GitHub repository without large datasets
