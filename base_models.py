
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Credit Card Default Prediction (UCI) veri setini yükle
default_of_credit_card_clients = fetch_ucirepo(id=350)
X_default = default_of_credit_card_clients.data.features
y_default = default_of_credit_card_clients.data.targets

# Credit Card Fraud Detection (Kaggle) veri setini yükle
df_fraud = pd.read_csv("creditcard.csv")
X_fraud = df_fraud.drop("Class", axis=1)
y_fraud = df_fraud["Class"]

# UCI veri seti için temel modelleri eğit
X_default_train, X_default_test, y_default_train, y_default_test = train_test_split(X_default, y_default, test_size=0.3, random_state=42)

# Logistic Regression
log_reg_default = LogisticRegression(max_iter=1000)
log_reg_default.fit(X_default_train, y_default_train.values.ravel())
y_pred_log_reg_default = log_reg_default.predict(X_default_test)
print("\nCredit Card Default Prediction - Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_default_test, y_pred_log_reg_default):.4f}")
print(classification_report(y_default_test, y_pred_log_reg_default))

# Decision Tree
dec_tree_default = DecisionTreeClassifier(random_state=42)
dec_tree_default.fit(X_default_train, y_default_train.values.ravel())
y_pred_dec_tree_default = dec_tree_default.predict(X_default_test)
print("\nCredit Card Default Prediction - Decision Tree:")
print(f"Accuracy: {accuracy_score(y_default_test, y_pred_dec_tree_default):.4f}")
print(classification_report(y_default_test, y_pred_dec_tree_default))

# Kaggle veri seti için temel modelleri eğit
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.3, random_state=42, stratify=y_fraud)

# Logistic Regression
log_reg_fraud = LogisticRegression(max_iter=1000)
log_reg_fraud.fit(X_fraud_train, y_fraud_train)
y_pred_log_reg_fraud = log_reg_fraud.predict(X_fraud_test)
print("\nCredit Card Fraud Detection - Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_fraud_test, y_pred_log_reg_fraud):.4f}")
print(classification_report(y_fraud_test, y_pred_log_reg_fraud))

# Decision Tree
dec_tree_fraud = DecisionTreeClassifier(random_state=42)
dec_tree_fraud.fit(X_fraud_train, y_fraud_train)
y_pred_dec_tree_fraud = dec_tree_fraud.predict(X_fraud_test)
print("\nCredit Card Fraud Detection - Decision Tree:")
print(f"Accuracy: {accuracy_score(y_fraud_test, y_pred_dec_tree_fraud):.4f}")
print(classification_report(y_fraud_test, y_pred_dec_tree_fraud))


