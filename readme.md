# Customer Segmentation & Churn Pattern Analytics in European Banking

## Live Demo & Project Video

### Streamlit Dashboard  
🔗 https://your-streamlit-link.streamlit.app/

### Project Demonstration Video  
🎥 https://youtu.be/your-video-link

### Research Report  
🔗 https://doi.org/your-doi-link

---

# Overview

This project presents an AI-driven banking analytics system designed for customer segmentation, churn prediction, financial risk analysis, and customer intelligence within European retail banking environments. The system combines exploratory data analysis, feature engineering, clustering techniques, PCA visualization, ensemble machine learning models, and predictive analytics to analyze customer behaviour and identify churn-risk patterns.

The project was developed under the Unified Mentor banking analytics project framework as part of an undergraduate applied machine learning and financial analytics study.

---

# Objectives

- Analyze customer behaviour using banking analytics
- Segment customers into meaningful behavioural groups using clustering techniques
- Predict customer churn using machine learning models
- Identify high-risk customer segments
- Build an AI-driven customer intelligence framework
- Develop an interactive Streamlit dashboard for visualization and prediction

---

# Dataset Information

A banking customer dataset containing 10,000 records was used to simulate:

- Customer demographics
- Financial behaviour
- Banking product usage
- Credit profiles
- Customer engagement patterns
- Geographic customer distribution
- Churn behaviour

The dataset includes customer information from:

- France
- Germany
- Spain

The project focuses on understanding how customer demographics, financial characteristics, and engagement patterns influence customer churn within banking systems.

---

# Technologies Used

## Programming Language

- Python

## Libraries & Frameworks

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib
- SciPy

---

# Exploratory Data Analysis (EDA)

The project includes detailed EDA involving:

- Customer churn distribution analysis
- Geography-wise churn analysis
- Gender-based churn analysis
- Age distribution analysis
- Active membership analysis
- Balance distribution analysis
- Salary analysis
- Credit score analysis
- Correlation heatmaps
- Outlier detection
- Financial behaviour analysis
- Customer engagement pattern analysis

---

# Feature Engineering

Several engineered features were created, including:

- AgeGroup
- CreditScoreCategory
- BalanceSegment
- SalarySegment
- TenureGroup
- HighValueCustomer
- ActivePremiumCustomer
- MultiProductUser
- LowCreditRisk

These engineered features improved customer profiling and machine learning model performance.

---

# Customer Segmentation

## Clustering Techniques Used

- K-Means Clustering
- PCA-based Visualization

## Evaluation Methods

- Elbow Method
- Silhouette Score Analysis

## Best Silhouette Score

- Approximately 0.41

The clustering pipeline helped identify meaningful customer segments such as:

- High-value customers
- Loyal customers
- Churn-risk customers
- Inactive customers
- Premium customers

---

# Machine Learning Models Evaluated

| Model | Accuracy |
|---|---|
| Random Forest | 86.45% |
| Gradient Boosting | 86.25% |
| XGBoost | 86.10% |
| Extra Trees | 86.00% |

---

# Best Performing Model

## Random Forest Classifier

- Accuracy: 86.45%
- Strong overall churn prediction capability
- Effective handling of complex customer behaviour patterns
- High business applicability for banking analytics systems

---

# Churn Prediction System

The prediction framework uses:

- Ensemble machine learning models
- Feature-engineered customer profiles
- Behavioural analytics
- Financial risk indicators
- Customer segmentation intelligence
- PCA-enhanced clustering insights

The system predicts:

- Churn probability
- Customer risk category
- Customer retention likelihood
- Financial risk exposure

---

# Streamlit Dashboard Features

The project includes an interactive Streamlit dashboard with:

- Executive Overview Dashboard
- Customer Churn Prediction System
- Customer Segmentation Visualization
- PCA Cluster Visualization
- Geography-wise Risk Analytics
- High-Value Customer Explorer
- Financial Risk Insights
- Multi-Model AI Prediction Center
- Interactive KPI Dashboard
- AI Insight Generator
- Executive Summary for Stakeholders
- Model Accuracy Comparison Charts

---

# Key Performance Indicators (KPIs)

- Overall Churn Rate
- Segment Churn Rate
- High-Value Churn Ratio
- Geographic Risk Index
- Engagement Drop Indicator
- Customer Retention Rate
- Active Customer Ratio

---

# Project Structure

```text
EU Bank/
│
├── Models/
│   ├── extra_trees.pkl
│   ├── gradient_boosting.pkl
│   ├── kmeans_model.pkl
│   ├── pca_model.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── xgboost.pkl
│
├── app.py
├── EU_Bank_Research_Paper.pdf
├── European_Bank.csv
├── European_Banking_Churn_Analytics.ipynb
├── Final_Banking_Churn_Dataset.csv
├── Government Stakeholder Executive Report.pdf
├── readme.md
└── requirements.txt
```

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

# Research & Academic Notes

- The dataset used in this project contains structured banking customer information.
- Results were evaluated within the available customer behavioural feature space.
- Clustering quality demonstrates moderate separation between customer groups.
- Machine learning performance depends on structured financial and engagement-related features.
- This work is intended as an educational, research-oriented, and banking analytics implementation project.

---

# Future Improvements

- Real-world banking dataset validation
- SHAP-based explainability analysis
- Deep learning integration
- Real-time banking analytics pipeline
- Live customer churn monitoring
- Cloud-based deployment architecture
- Advanced ensemble optimization
- Customer lifetime value prediction
- Explainable AI integration

---

# Author

## Adithya B V  
B.Sc. (Hons) Data Science and Analytics  
M.S. Ramaiah University of Applied Sciences

---

# Acknowledgement

This project was developed under the Unified Mentor banking analytics project framework.

---

# License

This project is released under MIT License.