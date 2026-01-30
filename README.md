# 📉 Customer Churn Prediction using Machine Learning + Streamlit

This project predicts whether a customer will **churn (leave the service)** based on telecom customer data.  
It includes **EDA, preprocessing, model training, evaluation, feature importance**, and a **Streamlit web app** for live churn prediction.

---


## 🌐 Live Demo
✅ Streamlit App Link: https://customer-churn-prediction-adan4su6jgbl4j4nlzpdge.streamlit.app/


## 🚀 Project Features
✅ Data Cleaning & Preprocessing  
✅ Exploratory Data Analysis (EDA)  
✅ Machine Learning Models:
- Logistic Regression
- Random Forest Classifier  
✅ Evaluation Metrics:
- Accuracy
- Recall
- ROC-AUC  
✅ Feature Importance Analysis  
✅ Streamlit Web App for prediction  
✅ Model saving using Joblib  

---

## 🧾 Dataset
- **Telco Customer Churn Dataset**
- Target column: **Churn** (Yes/No)

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## 📂 Project Structure



customer-Churn-Prediction/
│── churn_project.ipynb
│── README.md
│── requirements.txt
│
├── data/
│ └── Telco_Customer_Churn.csv
│
├── model/
│ ├── churn_model.pkl
│ ├── scaler.pkl
│ └── training_columns.pkl
│
└── app/
└── app.py


---

## ⚙️ How to Run the Project

### ✅ 1) Install Dependencies
```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
✅ 2) Run Jupyter Notebook

Open and run:

churn_project.ipynb

✅ 3) Run Streamlit Web App
python -m streamlit run app/app.py


Open in browser:
✅ http://localhost:8501

📊 Model Evaluation

The project is evaluated using:

Accuracy

Recall (important to catch churn customers)

ROC-AUC

Random Forest generally performs best and provides feature importance.

⭐ Feature Importance

The project identifies the most important churn-driving factors such as:

Contract type

Tenure

Monthly Charges

Payment Method

Internet Service


## ✅ Proof that project works
To run the project locally:

### 1) Install requirements
python -m pip install -r requirements.txt

### 2) Run Streamlit app
python -m streamlit run app/app.py

Then open:
http://localhost:8501



![image alt](https://github.com/Vedantsg23/Customer-Churn-Prediction/blob/b8cdddac048897ab4af638ba578cf6d67ec444de/Screenshot%202026-01-31%20004326.png)

✅ Author

Vedant Gadage
Computer Engineering Student



