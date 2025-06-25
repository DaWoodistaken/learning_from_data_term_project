
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Credit Card Default Prediction (UCI) veri setini yükle
default_of_credit_card_clients = fetch_ucirepo(id=350)
X_default = default_of_credit_card_clients.data.features
y_default = default_of_credit_card_clients.data.targets

# Credit Card Fraud Detection (Kaggle) veri setini yükle
df_fraud = pd.read_csv("creditcard.csv")
X_fraud = df_fraud.drop("Class", axis=1)
y_fraud = df_fraud["Class"]

# Veri setlerini eğitim ve test olarak ayır
X_default_train, X_default_test, y_default_train, y_default_test = train_test_split(X_default, y_default, test_size=0.3, random_state=42)
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.3, random_state=42, stratify=y_fraud)

# Temel Modelleri Eğit
# UCI Logistic Regression
log_reg_default = LogisticRegression(max_iter=1000)
log_reg_default.fit(X_default_train, y_default_train.values.ravel())
y_pred_log_reg_default = log_reg_default.predict(X_default_test)

# UCI Decision Tree
dec_tree_default = DecisionTreeClassifier(random_state=42)
dec_tree_default.fit(X_default_train, y_default_train.values.ravel())
y_pred_dec_tree_default = dec_tree_default.predict(X_default_test)

# Kaggle Logistic Regression
log_reg_fraud = LogisticRegression(max_iter=1000)
log_reg_fraud.fit(X_fraud_train, y_fraud_train)
y_pred_log_reg_fraud = log_reg_fraud.predict(X_fraud_test)

# Kaggle Decision Tree
dec_tree_fraud = DecisionTreeClassifier(random_state=42)
dec_tree_fraud.fit(X_fraud_train, y_fraud_train)
y_pred_dec_tree_fraud = dec_tree_fraud.predict(X_fraud_test)

# Öğrenmeme (Unlearning) için veri alt kümesi tanımlama
forget_indices_default = np.random.choice(X_default_train.index, size=int(len(X_default_train) * 0.1), replace=False)
X_default_forget = X_default_train.loc[forget_indices_default]
y_default_forget = y_default_train.loc[forget_indices_default]

X_default_retain = X_default_train.drop(forget_indices_default)
y_default_retain = y_default_train.drop(forget_indices_default)

forget_indices_fraud = np.random.choice(X_fraud_train.index, size=int(len(X_fraud_train) * 0.1), replace=False)
X_fraud_forget = X_fraud_train.loc[forget_indices_fraud]
y_fraud_forget = y_fraud_train.loc[forget_indices_fraud]

X_fraud_retain = X_fraud_train.drop(forget_indices_fraud)
y_fraud_retain = y_fraud_train.drop(forget_indices_fraud)

# Öğrenmeme Modellerini Eğit
student_model_default = LogisticRegression(max_iter=1000)
student_model_default.fit(X_default_retain, y_default_retain.values.ravel())
y_pred_unlearned_default = student_model_default.predict(X_default_test)

student_model_fraud = LogisticRegression(max_iter=1000)
student_model_fraud.fit(X_fraud_retain, y_fraud_retain)
y_pred_unlearned_fraud = student_model_fraud.predict(X_fraud_test)

# Sonuçları Değerlendirme ve Karşılaştırma
results = {
    'Dataset': [],
    'Model': [],
    'Accuracy': [],
    'Precision (Class 1)': [],
    'Recall (Class 1)': [],
    'F1-Score (Class 1)': []
}

def add_results(dataset, model_name, y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    results['Dataset'].append(dataset)
    results['Model'].append(model_name)
    results['Accuracy'].append(accuracy_score(y_true, y_pred))
    results['Precision (Class 1)'].append(report['1']['precision'] if '1' in report else 0)
    results['Recall (Class 1)'].append(report['1']['recall'] if '1' in report else 0)
    results['F1-Score (Class 1)'].append(report['1']['f1-score'] if '1' in report else 0)

# UCI Veri Seti Sonuçları
add_results('UCI', 'Logistic Regression (Base)', y_default_test, y_pred_log_reg_default)
add_results('UCI', 'Decision Tree (Base)', y_default_test, y_pred_dec_tree_default)
add_results('UCI', 'Logistic Regression (Unlearned)', y_default_test, y_pred_unlearned_default)

# Kaggle Veri Seti Sonuçları
add_results('Kaggle', 'Logistic Regression (Base)', y_fraud_test, y_pred_log_reg_fraud)
add_results('Kaggle', 'Decision Tree (Base)', y_fraud_test, y_pred_dec_tree_fraud)
add_results('Kaggle', 'Logistic Regression (Unlearned)', y_fraud_test, y_pred_unlearned_fraud)

results_df = pd.DataFrame(results)
print("\n--- Model Performance Comparison ---")
print(results_df)

# Görselleştirme: Doğruluk Karşılaştırması
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Accuracy', hue='Dataset', data=results_df)
plt.title('Model Accuracy Comparison')
plt.ylim(0.7, 1.0)
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('model_accuracy_comparison.png')
plt.close()

# Görselleştirme: Unutulan Veri Üzerindeki Performans
# UCI
y_pred_forget_default = student_model_default.predict(X_default_forget)
accuracy_forget_default = accuracy_score(y_default_forget, y_pred_forget_default)

# Kaggle
y_pred_forget_fraud = student_model_fraud.predict(X_fraud_forget)
accuracy_forget_fraud = accuracy_score(y_fraud_forget, y_pred_forget_fraud)

forget_data_results = pd.DataFrame({
    'Dataset': ['UCI', 'Kaggle'],
    'Accuracy on Forgotten Data': [accuracy_forget_default, accuracy_forget_fraud]
})

plt.figure(figsize=(8, 5))
sns.barplot(x='Dataset', y='Accuracy on Forgotten Data', data=forget_data_results)
plt.title('Unlearned Model Performance on Forgotten Data')
plt.ylim(0.0, 1.0)
plt.ylabel('Accuracy')
plt.savefig('unlearned_performance_on_forgotten_data.png')
plt.close()


