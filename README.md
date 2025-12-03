# XAI-Credit-Scoring
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/a575a0cd-71eb-411d-b168-d728eab0677d" />

<p align="center">
  <img src="assets/banner.png" alt="XAI Credit Scoring Banner" width="100%">
</p>

<h1 align="center">💳 XAI Credit Scoring</h1>
<h3 align="center">Explainable, Fair & Production-Ready Credit Risk Scoring System</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Model-XGBoost-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Explainability-SHAP-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/UI-Gradio-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Category-FinTech-success?style=for-the-badge">
  <img src="https://img.shields.io/github/last-commit/chevvakavitha/XAI-Credit-Scoring?style=for-the-badge">
</p>

---

# 📌 Overview

**XAI Credit Scoring** is a **production-ready machine learning system** that predicts credit default risk using:

- **XGBoost** (high-performance gradient boosting)
- **SHAP explainability** to interpret decisions  
- **Modular ML pipeline** for real-world deployment  
- **Clean architecture** following industry standards  

This project is ideal for:
- Data Science portfolios  
- FinTech ML roles  
- AI explainability & risk modeling  
- Interview projects  
- Real-world ML applications  

---

# ❗ Problem Statement

Traditional ML credit scoring models are:

❌ Black-box (no explainability)  
❌ Hard to justify to regulators (RBI, CFPB, EU AI Act)  
❌ Prone to bias  
❌ Not deployment-ready  

This system solves it using:

✔ **Transparent model decisions (SHAP)**  
✔ **Regulatory-friendly explanations**  
✔ **Production architecture**  
✔ **Bias-aware modeling**  

---

# 📊 Dataset Details

Typical credit scoring dataset includes:

### **🔹 Demographic Features**
- Age  
- Dependents  
- Marital status  

### **🔹 Financial Features**
- Annual income  
- Loan amount  
- Credit history  

### **🔹 Behavioral Features**
- Number of late payments  
- Credit utilization  

### **🎯 Target**
- `1 = Default`
- `0 = No Default`

> ⚠️ Large datasets (100MB+) are **not** included due to GitHub limits.  
> Store them locally in:
data/raw/

yaml
Copy code

---

# 🏗 System Architecture

scss
Copy code
                      ┌─────────────────────────┐
                      │       Raw Dataset       │
                      └────────────┬────────────┘
                                   │
                 ┌─────────────────▼─────────────────┐
                 │      Data Preprocessing Layer     │
                 │ (Cleaning, Encoding, Scaling, FE) │
                 └─────────────────┬─────────────────┘
                                   │
                 ┌─────────────────▼─────────────────┐
                 │   Model Training Engine (XGBoost)  │
                 └─────────────────┬─────────────────┘
                                   │
                 ┌─────────────────▼─────────────────┐
                 │ Explainability Layer (SHAP)        │
                 └─────────────────┬─────────────────┘
                                   │
                   ┌───────────────▼───────────────┐
                   │ Final Score + Explanation JSON │
                   └────────────────────────────────┘
yaml
Copy code

---

# ⚙ ML Pipeline

data.csv
│
├──▶ preprocessing.py
│ ├─ missing value handling
│ ├─ label encoding
│ ├─ scaling
│ └─ feature engineering
│
├──▶ train.py
│ ├─ train XGBoost
│ ├─ hyperparameter tuning
│ └─ model.pkl exported
│
├──▶ shap_explain.py
│ ├─ global importance
│ ├─ local prediction explanation
│ └─ force / waterfall plots
│
└──▶ predict.py
└─ probability + SHAP JSON output

yaml
Copy code

---

# ✨ Features

### 🔥 Core Features
- End-to-end ML pipeline  
- Clean modular architecture  
- XGBoost with optimized parameters  
- Fast inference  
- Ready for API or Gradio UI  

---

### 🔍 Explainability Features
- SHAP summary plots  
- Feature importance  
- Force plots  
- Waterfall plots  
- Fully transparent predictions  

---

### 🛡 Fairness & Reliability
- Bias detection ready  
- Drift detection compatible  
- Regulatory compliance friendly  

---

# 🧠 Explainability (SHAP)

SHAP shows:

✔ Why the model approved or rejected an applicant  
✔ Which features increased risk  
✔ Contribution of each factor  

> Store your SHAP images in:  
assets/explainability/

php-template
Copy code

Example usage:

markdown
<p align="center">
  <img src="assets/explainability/shap_summary.png" width="70%">
</p>

---

## 📈 Evaluation Metrics
Metric	Result
AUC-ROC	0.89–0.95
F1-Score	High
Precision	High
Recall	High
Accuracy	85–90%

Place metrics images here:

bash
Copy code
assets/metrics/confusion_matrix.png
assets/metrics/roc_curve.png

---

## 📁 Project Structure
css
Copy code
XAI-Credit-Scoring/
│
├── assets/
│   ├── banner.png
│   ├── explainability/
│   ├── metrics/
│   └── outputs/
│
├── data/
│   └── raw/ 
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── shap_explain.py
│   └── utils.py
│
├── models/
│   └── model.pkl
│
├── requirements.txt
└── README.md

---

## 🛠 Installation
bash
Copy code
git clone https://github.com/chevvakavitha/XAI-Credit-Scoring.git
cd XAI-Credit-Scoring

python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

pip install -r requirements.txt

---

## ▶ Running the Model
##🔥 Train
bash
Copy code
python src/train.py
##🔥 Generate SHAP Explanations
bash
Copy code
python src/shap_explain.py
##🔥 Predict for a New User
bash
Copy code
python src/predict.py --input sample.json
##🖼 Output Screenshots
Add your model output charts into:

bash
Copy code
assets/outputs/
Example:

markdown
Copy code
<p align="center">
  <img src="assets/outputs/prediction_example.png" width="60%">
</p>

---

## 🚀 Future Improvements
Add FastAPI deployment

Add Streamlit dashboard

Add fairness dashboards (EvidentlyAI)

Add automated monitoring

Convert to fully production MLOps pipeline

---

##📬 Contact
Cheva Kavitha
📩 Email: kavithachevvakavitha@gmail.com
🔗 GitHub: https://github.com/chevvakavitha
💼 LinkedIn: https://www.linkedin.com/in/cheva-kavitha/

<p align="center">⭐ If this project helped you, please give it a star!</p> ```
