# MediVision AI — Notebook Guide

> **File:** `notebooks/01_MediVision_Predictive_Health_Analytics.ipynb`  
> **Dataset:** NHANES Cycle P (2017–2020) · CDC, United States  
> **Goal:** Predict 6 chronic disease risks from real patient data using a clean, reproducible ML pipeline.

---

## Table of Contents

1. [What This Notebook Does](#1-what-this-notebook-does)
2. [Data Source](#2-data-source)
3. [Section 1 — Setup](#3-section-1--setup)
4. [Section 2 — Download Data](#4-section-2--download-data)
5. [Section 3 — Load & Clean Data](#5-section-3--load--clean-data)
6. [Section 4 — Exploratory Data Analysis (EDA)](#6-section-4--exploratory-data-analysis-eda)
7. [Section 5 — Train Disease Prediction Models](#7-section-5--train-disease-prediction-models)
8. [Section 6 — Patient Risk Report](#8-section-6--patient-risk-report)
9. [Section 7 — Save Models for Backend API](#9-section-7--save-models-for-backend-api)
10. [Evaluation Metrics Explained](#10-evaluation-metrics-explained)
11. [Glossary](#11-glossary)

---

## 1. What This Notebook Does

This notebook builds an end-to-end machine learning pipeline that:

- Downloads real US health survey data (NHANES) automatically
- Cleans and merges 12 data files into one analysis-ready table
- Explores patterns with charts and statistics
- Trains **one machine learning model per disease** — 6 models total
- Predicts risk for 6 chronic diseases at once

**Diseases predicted:**

| # | Disease | Positive Rate |
|---|---------|:------------:|
| 1 | Diabetes | ~15% |
| 2 | Hypertension | ~37% |
| 3 | Heart Disease | ~8% |
| 4 | Asthma | ~16% |
| 5 | Stroke | ~5% |
| 6 | Arthritis | ~31% |

---

## 2. Data Source

**NHANES** = National Health and Nutrition Examination Survey  
Published by the **CDC (Centers for Disease Control and Prevention)**, USA.  
Cycle P covers **August 2017 – March 2020** (pre-pandemic).

Each respondent has a unique ID (`SEQN`) linking records across all survey modules.

### Files Used

| File | Topic | Key columns |
|------|-------|-------------|
| `P_DEMO.xpt` | Demographics | Age, gender, ethnicity, education, income |
| `P_BMX.xpt` | Body measurements | Height, weight, BMI, waist circumference |
| `P_BPQ.xpt` | Blood pressure questions | Hypertension diagnosis |
| `P_DIQ.xpt` | Diabetes questions | Diabetes diagnosis |
| `P_GHB.xpt` | Glycohaemoglobin | HbA1c (3-month blood sugar average) |
| `P_GLU.xpt` | Plasma glucose | Fasting blood glucose |
| `P_HDL.xpt` | HDL cholesterol | Good cholesterol |
| `P_MCQ.xpt` | Medical conditions | Heart disease, asthma, stroke, arthritis |
| `P_PAQ.xpt` | Physical activity | Sedentary minutes per day |
| `P_SMQ.xpt` | Smoking | Ever smoked |
| `P_TCHOL.xpt` | Total cholesterol | Total blood cholesterol |
| `P_TRIGLY.xpt` | Triglycerides | Blood fat + calculated LDL |

---

## 3. Section 1 — Setup

**What it does:** Imports all required Python libraries in one place.

**Libraries used:**

| Library | Purpose |
|---------|---------|
| `numpy`, `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Charts and visualisations |
| `sklearn` | Machine learning models and evaluation |
| `requests` | Downloading data files |
| `joblib` | Model serialization |

---

## 4. Section 2 — Download Data

**What it does:** Downloads each NHANES `.xpt` file from CDC servers if it doesn't already exist locally.

Files are stored in `data/raw/`. The download is skipped if the file is already present — so running the notebook twice won't re-download anything.

---

## 5. Section 3 — Load & Clean Data

Five steps to prepare the data:

1. **Load and merge**: All 12 files are joined into one table using a left join on `SEQN`.
2. **Rename columns**: CDC variable codes (e.g. `BMXBMI`, `LBXGH`) are renamed to plain English.
3. **Handle Categories**: Converting numerical codes for demographics into meaningful inputs.
4. **Adults Only**: Only respondents aged **18+** are kept to reduce noise for adult-onset chronic diseases.
5. **Disease Targets**: Each disease gets a binary column (`1` = has disease, `0` = does not).

> Heart disease combines **Coronary Heart Disease**, **Angina**, and **Heart Attack** into one positive label.

---

## 6. Section 4 — Exploratory Data Analysis (EDA)

Before building any model, we explore the data to understand distributions, missing values, and feature relationships.

### 4.1 to 4.4: Basic Summaries
- Sets up visual colors and counts total rows, columns, and memory usage.
- Counts missing values (especially in blood-test columns, which are often 40-60% missing due to fasting requirements).
- Calculates mean, median, standard deviation, and skewness for numeric features.

### 4.5 How Many People Have Each Disease?
Bar charts showing positive vs. negative counts for each disease. Imbalance ratios are highlighted (e.g., a 10:1 ratio means 10 healthy patients for every 1 sick patient).

### 4.6 Feature Distributions
Histograms for every numeric feature showing the mean (red) and median (orange).

### 4.7 Categorical Feature Counts
Bar charts showing patient breakdowns across categories like education, gender, and ethnicity.

### 4.9 Correlation Between Features
A heatmap identifying strong relationships between numeric features (e.g., BMI and weight, or HbA1c and fasting glucose).

### 4.11 Outliers
We flag outliers using the IQR method. 
> **Note:** Medical outliers represent real extreme cases (e.g., HbA1c > 14% in uncontrolled diabetes). They are **preserved**, not removed, because they carry the strongest disease signal.

### 4.12 Do Diseases Appear Together?
Measures co-morbidities using the **Phi coefficient** (correlation between binary disease targets).

### 4.13 EDA Summary
Prints a quick summary of key findings regarding imbalances, strong correlations, and missing data.

---

## 7. Section 5 — Train Disease Prediction Models

We train **one model per disease** using `HistGradientBoostingClassifier`.

**Why this algorithm?**
- **Handles `NaN` natively:** 40–60% of lab values are missing — no imputation needed.
- **Histogram binning:** Fast training on 9,000+ patients.
- **`class_weight='balanced'`:** Automatically corrects for imbalanced disease rates.

### Overfitting Prevention
Each disease gets individual hyperparameters to prevent the model from memorising the training data:
- `max_depth` reduced for shallower trees.
- `min_samples_leaf` increased so splits require more data.
- `l2_regularization` increased.
- `early_stopping=True` for diseases prone to overfitting (Heart Disease, Asthma, Stroke, Arthritis).

### 5.2 Model Evaluation
Evaluates every model with:
- **ROC Curve:** True Positive Rate vs. False Positive Rate.
- **Precision-Recall Curve:** Highly informative for rare diseases (like Stroke).
- **Confusion Matrix:** Counts of correct and incorrect predictions.
- **Combined ROC Chart:** All 6 diseases on a single chart for comparison.
- **Summary Table:** A final printout comparing train/test AUC across all diseases to ensure overfitting was resolved.

---

## 8. Section 6 — Patient Risk Report

The `patient_report()` function runs all 6 trained models on a single patient's measurements and prints a unified risk profile.

**Example output:**
```text
[ PATIENT 1 ]
Patient: Age 55, BMI 23.6
---------------------------------------------
  Diabetes        Low Risk    1.5%  [                    ]
  Hypertension    Low Risk   17.4%  [###                 ]
  Heart Disease   Low Risk   15.8%  [###                 ]
  Asthma          Low Risk   44.1%  [########            ]
  Stroke          Low Risk   31.3%  [######              ]
  Arthritis       AT RISK    51.1%  [##########          ]
---------------------------------------------
```

---

## 9. Section 7 — Save Models for Backend API

Saves the `trained_models` and `selected_features` lists as `.joblib` files into the `models/` directory, allowing the FastAPI backend to serve predictions.

---

## 10. Evaluation Metrics Explained

| Metric | Formula | When to use |
|--------|---------|-------------|
| **Accuracy** | (TP+TN) / Total | Only when classes are balanced |
| **Precision** | TP / (TP+FP) | When false alarms are costly |
| **Recall** | TP / (TP+FN) | When missing sick patients is costly |
| **F1** | 2 × P×R / (P+R) | Balance between Precision and Recall |
| **ROC-AUC** | Area under ROC | Overall ranking ability, robust to imbalance |
| **PR-AUC** | Area under PR | Best for rare positive class |

> In a clinical context, **Recall** is usually the most important metric — missing a sick patient (False Negative) is more harmful than a false alarm (False Positive).

---

## 11. Glossary

| Term | Definition |
|------|-----------|
| **AUC** | Area Under the Curve — summarises ROC or PR curve in one number |
| **BMI** | Body Mass Index = weight(kg) / height(m)² |
| **Class imbalance** | One class (sick) is much rarer than the other (healthy) |
| **Confusion Matrix** | Table of TP, TN, FP, FN counts |
| **Early Stopping** | Halts training when validation performance stops improving |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **False Negative** | Sick patient predicted as healthy — most dangerous error |
| **False Positive** | Healthy patient predicted as sick — leads to unnecessary tests |
| **HbA1c** | Glycohaemoglobin — reflects average blood sugar over 3 months |
| **HDL** | High-Density Lipoprotein — "good" cholesterol |
| **IQR** | Interquartile Range = Q3 − Q1 |
| **kurtosis** | Heaviness of distribution tails |
| **LDL** | Low-Density Lipoprotein — "bad" cholesterol |
| **NHANES** | National Health and Nutrition Examination Survey (CDC) |
| **Overfit Delta** | Train AUC − Test AUC; measures how much a model memorised training data |
| **Phi coefficient (φ)** | Correlation between two binary (yes/no) variables |
| **PR Curve** | Precision-Recall curve — useful when positive class is rare |
| **Recall** | Proportion of actual positive cases the model correctly identifies |
| **ROC Curve** | Receiver Operating Characteristic — Recall vs. False Alarm Rate |
| **SEQN** | Unique respondent ID in NHANES, used to join all data modules |
| **Skewness** | Asymmetry of a distribution (0 = symmetric) |
| **Stratified split** | Train/test split that preserves the original class ratio |
| **XPT** | SAS Transport format used by CDC for NHANES data distribution |
