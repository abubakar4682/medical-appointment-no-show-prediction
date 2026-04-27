## 🚀 Live Demo

[![Open App](https://img.shields.io/badge/Streamlit-Live_App-brightgreen?logo=streamlit)](https://medical-appointment-no-show-prediction-h5gtnnwg2uua2qyssxvya3.streamlit.app/)
> **⚠️ Educational project only — not intended for real clinical or medical decision-making.**

A full end-to-end data science and machine learning portfolio project that predicts whether a patient is likely to miss a hospital appointment, built with Python and deployed via Streamlit.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology (CRISP-DM)](#methodology-crisp-dm)
- [Key Findings](#key-findings)
- [Model Comparison](#model-comparison)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Ethics & Fairness](#ethics--fairness)
- [Project Info](#project-info)

---

## 📌 Project Overview

This project tackles the real-world problem of patient **no-shows** in public hospital settings. Using the **Medical Appointment No-Show dataset** (110,527 records from Brazil), it applies a complete data science workflow:

- Data cleaning & feature engineering
- Exploratory data analysis (EDA)
- Machine learning modelling (Logistic Regression, Random Forest, XGBoost)
- Model evaluation & selection
- Deployment as an interactive Streamlit web app

**Prediction Target:**
| Value | Meaning |
|-------|---------|
| `0` | Patient attended the appointment |
| `1` | Patient missed the appointment |

---

## 📊 Dataset

The dataset contains appointment records from public hospitals in Brazil.

| Feature | Description |
|---------|-------------|
| `Age` | Age of the patient |
| `Gender` | Patient gender |
| `ScheduledDay` | Date the appointment was booked |
| `AppointmentDay` | Date of the actual appointment |
| `Scholarship` | Whether the patient is in a welfare programme |
| `Hypertension` | Whether the patient has hypertension |
| `Diabetes` | Whether the patient has diabetes |
| `Alcoholism` | Whether the patient has alcoholism |
| `Handicap` | Disability indicator |
| `SMS_received` | Whether the patient received an SMS reminder |
| `No-show` | Target variable (attended or missed) |

**Engineered Features:**
- `wait_days` — Number of days between scheduling and appointment date
- `age_group` — Categorical grouping: Child, Teen, Young Adult, Adult, Senior

---

## 🧠 Methodology (CRISP-DM)

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

| Stage | What Was Done |
|-------|--------------|
| **Business Understanding** | Defined the problem: predict patient no-shows |
| **Data Understanding** | Explored structure, distributions, missing values, and target imbalance |
| **Data Preparation** | Cleaned columns, removed invalid values, converted dates, engineered features |
| **Modelling** | Trained Logistic Regression, Random Forest, and XGBoost |
| **Evaluation** | Compared models on accuracy, recall, F1-score, and confusion matrices |
| **Deployment** | Built an interactive Streamlit app for real-time prediction |

---

## 📈 Key Findings

- Patients with **longer waiting times** were more likely to miss appointments.
- **Teenagers and young adults** showed higher no-show rates than older patients.
- **SMS reminders** correlated with a higher no-show rate — likely due to selection bias (reminders sent to higher-risk groups).
- **`wait_days`** was the most important predictor across all models.

---

## 🤖 Model Comparison

| Model | Accuracy | Recall (No-show) | F1-score (No-show) |
|-------|----------|-----------------|-------------------|
| Logistic Regression | 79% | 0.02 | 0.03 |
| **Random Forest** ✅ | **77%** | **0.22** | **0.29** |
| XGBoost | 80% | 0.05 | 0.09 |

> **Selected Model: Random Forest** — Although XGBoost achieved the highest accuracy, Random Forest was selected for its superior recall and F1-score on the no-show class. In this problem, correctly identifying missed appointments is more valuable than raw accuracy.

---

## 📁 Project Structure

```
appointment-noshow-project/
│
├── app.py                  # Streamlit web application
├── rf_model.pkl            # Trained Random Forest model
├── feature_columns.pkl     # Saved feature column list
├── requirements.txt        # Python dependencies
│
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks (EDA, modelling)
├── reports/                # Analysis reports and outputs
└── images/                 # Visualisations and screenshots
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/appointment-noshow-project.git
cd appointment-noshow-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### Dependencies

```
streamlit | pandas | numpy | scikit-learn | joblib | xgboost | matplotlib | seaborn
```

---

## ⚖️ Ethics & Fairness

This project considers key ethical principles in healthcare AI:

| Concern | Consideration |
|---------|---------------|
| **Privacy** | Patient data should be anonymised and handled securely |
| **GDPR** | Personal data must be processed lawfully and responsibly |
| **Fairness** | The model must not unfairly disadvantage specific demographic groups |
| **Automation Bias** | Model predictions should support — not replace — human clinical judgement |
| **Healthcare Risk** | False predictions could affect patient care; human oversight is essential |

---

## 📝 Project Info

| Field | Detail |
|-------|--------|
| **Module** | COM747 — Data Science and Machine Learning |
| **Project Type** | End-to-end ML portfolio project |
| **Framework** | CRISP-DM |
| **Deployment** | Streamlit Community Cloud |
| **Primary Tool** | Streamlit |
