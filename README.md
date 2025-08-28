# Credit Card Fraud Detection (Streamlit App)

A machine learning project to detect fraudulent credit card transactions using advanced models like Logistic Regression, Random Forest, and XGBoost.  
The project includes both data analysis & a deployed **Streamlit app** where users can test fraud predictions.

## 🚀 Features
- Exploratory Data Analysis (EDA) with visual insights.  
- Baseline models (Logistic Regression, Random Forest, XGBoost).  
- Advanced modeling with imbalance handling (SMOTE, class weights, scale_pos_weight).  
- Model evaluation using Precision, Recall, F1, ROC-AUC, PR-AUC.  
- **Streamlit app**:
  - Upload your own CSV or use a default sample.
  - Get fraud probability per transaction.
  - Interactive Precision-Recall curve.

## 🛠️ Tech Stack
- Python (pandas, scikit-learn, XGBoost, imbalanced-learn)  
- Plotly (visualizations)  
- Streamlit (deployment)  

## 📂 How to Run Locally
```bash
git clone https://github.com/your-username/fraud-detection-streamlit.git
cd fraud-detection-streamlit
pip install -r requirements.txt
streamlit run app/app.py


# creditcard-fraud-detection-model
