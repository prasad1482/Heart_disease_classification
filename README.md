🩺 Heart Disease Prediction using Machine Learning
🔍 Objective
The goal of this project is to build and compare machine learning models that can accurately predict the presence of heart disease based on patient data such as age, blood pressure, cholesterol, and more.

📁 Dataset
The dataset is publicly available and contains:

No missing values

Only numerical features (no categorical variables)

Target Column: target (1 = presence of heart disease, 0 = no disease)

🧪 Techniques Used
Step	Description
EDA	Exploratory analysis to understand data distribution
Preprocessing	Scaling features using StandardScaler
Train/Test Split	80/20 split using train_test_split
Models Used	Logistic Regression, Random Forest
Evaluation Metrics	Accuracy, Confusion Matrix, Precision, Recall, F1-score
Cross-Validation	5-Fold CV for robust model comparison
Visualization	Performance comparison using bar graphs with error bars

📊 Model Performance
Logistic Regression
Accuracy: ~79.5%

Precision (Class 1): 0.76

Recall (Class 1): 0.87

F1-score (Class 1): 0.81

Random Forest
Accuracy: ~98.5%

Precision (Class 1): 1.00

Recall (Class 1): 0.97

F1-score (Class 1): 0.99

✅ Random Forest outperformed Logistic Regression in all metrics and showed excellent generalization.

📈 Visual Comparison

Bar chart showing cross-validation accuracy with error bars for both models.

🚀 Future Improvements
Add more models (e.g., XGBoost, SVM)

Hyperparameter tuning using GridSearchCV

Model deployment using Streamlit or Flask

Explore SHAP/LIME for model interpretability

📂 How to Run
bash
Copy
Edit
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
python heart_disease_model.py
🙌 Acknowledgements
Dataset sourced from UCI Machine Learning Repository

Tools used: scikit-learn, matplotlib, seaborn, pandas, numpy

