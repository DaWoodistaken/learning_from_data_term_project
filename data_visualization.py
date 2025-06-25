
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Credit Card Default Prediction (UCI) veri setini yükle
default_of_credit_card_clients = fetch_ucirepo(id=350)
X_default = default_of_credit_card_clients.data.features
y_default = default_of_credit_card_clients.data.targets

# Credit Card Fraud Detection (Kaggle) veri setini yükle
df_fraud = pd.read_csv("creditcard.csv")

# Credit Card Default Prediction (UCI) veri seti görselleştirmeleri

# Y (default payment) dağılımı
plt.figure(figsize=(6, 4))
sns.countplot(x=y_default["Y"])
plt.title("Credit Card Default Payment Distribution")
plt.xlabel("Default Payment (1=Yes, 0=No)")
plt.ylabel("Count")
plt.savefig("default_payment_distribution.png")
plt.close()

# X1 (credit amount) dağılımı
plt.figure(figsize=(8, 6))
sns.histplot(X_default["X1"], bins=50, kde=True)
plt.title("Credit Amount Distribution")
plt.xlabel("Credit Amount (NT dollar)")
plt.ylabel("Frequency")
plt.savefig("credit_amount_distribution.png")
plt.close()

# X5 (age) dağılımı
plt.figure(figsize=(8, 6))
sns.histplot(X_default["X5"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age (year)")
plt.ylabel("Frequency")
plt.savefig("age_distribution.png")
plt.close()

# X2 (gender) dağılımı
plt.figure(figsize=(6, 4))
sns.countplot(x=X_default["X2"])
plt.title("Gender Distribution")
plt.xlabel("Gender (1=Male, 2=Female)")
plt.ylabel("Count")
plt.savefig("gender_distribution.png")
plt.close()

# Credit Card Fraud Detection (Kaggle) veri seti görselleştirmeleri

# Class (fraudulent or genuine) dağılımı
plt.figure(figsize=(6, 4))
sns.countplot(x=df_fraud["Class"])
plt.title("Credit Card Fraud Detection - Class Distribution")
plt.xlabel("Class (0=Genuine, 1=Fraud)")
plt.ylabel("Count")
plt.savefig("fraud_class_distribution.png")
plt.close()

# Amount dağılımı
plt.figure(figsize=(8, 6))
sns.histplot(df_fraud["Amount"], bins=50, kde=True)
plt.title("Credit Card Fraud Detection - Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.savefig("fraud_amount_distribution.png")
plt.close()

# Time dağılımı
plt.figure(figsize=(8, 6))
sns.histplot(df_fraud["Time"], bins=50, kde=True)
plt.title("Credit Card Fraud Detection - Time Distribution")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency")
plt.savefig("fraud_time_distribution.png")
plt.close()


