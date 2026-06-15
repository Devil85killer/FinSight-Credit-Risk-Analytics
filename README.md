# FinSight: Predictive Credit Risk Analytics & Dashboard

![FinSight Dashboard](Screenshot%202026-06-15%20163433.png)

## 📌 Project Overview
Financial institutions lose millions annually to credit defaults due to static, outdated risk assessment rules. **FinSight** is an end-to-end predictive analytics pipeline designed to identify high-risk credit customers before they default. 

This project integrates a Python-based Machine Learning backend with an interactive Power BI executive dashboard to dynamically score and visualize $19M in financial exposure across a 10,000+ customer portfolio.

## 🛠️ Tech Stack & Tools
* **Data Engineering & ML:** Python (Pandas, NumPy, Scikit-Learn, Imbalanced-Learn)
* **Business Intelligence:** Power BI, DAX
* **Key Techniques:** SMOTE (Synthetic Minority Over-sampling Technique), Random Forest Classification, Exploratory Data Analysis (EDA), KPI Development

## 🚀 Methodology & Workflow

### 1. Data Processing & Machine Learning (Python)
* Ingested and cleaned a dataset of 10,000+ credit customers, engineering behavioral features such as `Credit_Utilization_Ratio` and `Number_of_Late_Payments`.
* Identified extreme class imbalance in the default data and applied **SMOTE** to synthesize minority class samples, increasing model recall for high-risk profiles by 100%.
* Trained a **Random Forest Classifier** to predict the probability of default, extracting these probabilities to assign a dynamic `Risk_Score` (1-100) to every customer.

### 2. Enterprise Visualization & BI (Power BI)
* Imported the scored dataset into Power BI and established an enterprise-grade dark-theme reporting layout.
* Developed custom **DAX measures** (using `CALCULATE`, `IF`, and nested logic) to aggregate high-risk metrics dynamically.
* Designed an interactive, 10-element executive dashboard featuring:
  * **Financial KPIs:** Tracking total high-risk customers, $19M in total risk exposure, and fraud incidents.
  * **Behavioral Scatter Plots:** Visualizing the correlation between credit utilization and credit scores.
  * **Operational Hit-Lists:** A dynamically filtered matrix highlighting specific Suspicious IDs for immediate intervention.

## 📊 Business Impact
The implementation of the FinSight pipeline allows stakeholders to transition from reactive penalty enforcement to proactive risk mitigation, providing instant visibility into demographic default trends and specific high-risk account exposures.
