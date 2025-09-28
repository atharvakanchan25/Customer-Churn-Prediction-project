# Customer Churn Prediction Dashboard

## Project Overview
This project predicts customer churn for a telecom company using machine learning and visualizes insights through an interactive Power BI dashboard. The aim is to identify customers at risk of leaving and provide actionable insights for business decisions.

---

## Tools & Technologies
- **Python**: pandas, scikit-learn, matplotlib, seaborn, joblib  
- **Machine Learning Model**: Random Forest Classifier  
- **Power BI**: Dashboard creation & interactive visualizations  
- **Version Control**: Git & GitHub  

---

## Folder Structure

Customer-Churn-Prediction/
│
├── data/
│ ├── raw/ # Original dataset
│ │ └── Telco-Customer-Churn.csv
│ └── processed/ # Cleaned dataset
│ └── customer_churn_cleaned.csv
│
├── scripts/ # Python scripts
│ ├── preprocess.py # Data cleaning & preprocessing
│ ├── train_model.py # Model training
│ ├── evaluate_model.py # Model evaluation
│ └── eda_visualizations.py # Exploratory Data Analysis & plots
│
├── outputs/
│ ├── figures/ # Plots for EDA & feature importance
│ ├── models/ # Trained ML model
│ └── reports/ # Evaluation reports
│
├── notebooks/ # Optional Jupyter notebooks
│ ├── 01_data_exploration.ipynb
│ ├── 02_preprocessing.ipynb
│ └── 03_model_building.ipynb
│
├── powerbi_dashboard/ # Power BI dashboard screenshots
├── utils/ # Helper functions
├── requirements.txt
└── README.md