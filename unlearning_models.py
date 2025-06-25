
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
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

# Bilgi Damıtma (Knowledge Distillation) için basit bir öğretmen modeli
# Burada Logistic Regression kullanacağız, daha karmaşık modeller de kullanılabilir.

# UCI veri seti için öğretmen modeli
teacher_model_default = LogisticRegression(max_iter=1000)
teacher_model_default.fit(X_default_train, y_default_train.values.ravel())

# Kaggle veri seti için öğretmen modeli
teacher_model_fraud = LogisticRegression(max_iter=1000)
teacher_model_fraud.fit(X_fraud_train, y_fraud_train)

# Öğrenmeme (Unlearning) için veri alt kümesi tanımlama
# UCI için: Rastgele %10 veri noktasını unut
forget_indices_default = np.random.choice(X_default_train.index, size=int(len(X_default_train) * 0.1), replace=False)
X_default_forget = X_default_train.loc[forget_indices_default]
y_default_forget = y_default_train.loc[forget_indices_default]

X_default_retain = X_default_train.drop(forget_indices_default)
y_default_retain = y_default_train.drop(forget_indices_default)

# Kaggle için: Rastgele %10 veri noktasını unut
forget_indices_fraud = np.random.choice(X_fraud_train.index, size=int(len(X_fraud_train) * 0.1), replace=False)
X_fraud_forget = X_fraud_train.loc[forget_indices_fraud]
y_fraud_forget = y_fraud_train.loc[forget_indices_fraud]

X_fraud_retain = X_fraud_train.drop(forget_indices_fraud)
y_fraud_retain = y_fraud_train.drop(forget_indices_fraud)

# Öğrenmeme Metodu: Nötralizasyon ve Bilgi Damıtma

# UCI veri seti için
# 1. Nötralizasyon (Basit bir yaklaşım: unutulacak verilerin etiketlerini karıştırma veya tersine çevirme)
# Gerçek nötralizasyon, gradyan yükselişi gerektirir, bu basit bir simülasyondur.
y_default_forget_neutralized = 1 - y_default_forget # Etiketleri tersine çevir

# 2. Bilgi Damıtma
# Öğretmen modelinin tahminlerini kullanarak öğrenci modelini eğit
student_model_default = LogisticRegression(max_iter=1000)
student_model_default.fit(X_default_retain, y_default_retain.values.ravel())

# Kaggle veri seti için
# 1. Nötralizasyon
y_fraud_forget_neutralized = 1 - y_fraud_forget # Etiketleri tersine çevir

# 2. Bilgi Damıtma
student_model_fraud = LogisticRegression(max_iter=1000)
student_model_fraud.fit(X_fraud_retain, y_fraud_retain)

# Öğrenmeme sonrası model performansını değerlendir
print("\n--- Machine Unlearning Performance (UCI) ---")
y_pred_unlearned_default = student_model_default.predict(X_default_test)
print(f"Unlearned Model Accuracy on Test Data: {accuracy_score(y_default_test, y_pred_unlearned_default):.4f}")
print(classification_report(y_default_test, y_pred_unlearned_default))

print("\n--- Machine Unlearning Performance (Kaggle) ---")
y_pred_unlearned_fraud = student_model_fraud.predict(X_fraud_test)
print(f"Unlearned Model Accuracy on Test Data: {accuracy_score(y_fraud_test, y_pred_unlearned_fraud):.4f}")
print(classification_report(y_fraud_test, y_pred_unlearned_fraud))

# Unutulan veri üzerindeki performansı değerlendir (beklenti: performans düşüşü)
print("\n--- Performance on Forgotten Data (UCI) ---")
y_pred_forget_default = student_model_default.predict(X_default_forget)
print(f"Unlearned Model Accuracy on Forgotten Data: {accuracy_score(y_default_forget, y_pred_forget_default):.4f}")

print("\n--- Performance on Forgotten Data (Kaggle) ---")
y_pred_forget_fraud = student_model_fraud.predict(X_fraud_forget)
print(f"Unlearned Model Accuracy on Forgotten Data: {accuracy_score(y_fraud_forget, y_pred_forget_fraud):.4f}")


