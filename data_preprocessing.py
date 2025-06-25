
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Credit Card Default Prediction (UCI) veri setini yükle
default_of_credit_card_clients = fetch_ucirepo(id=350)
X_default = default_of_credit_card_clients.data.features
y_default = default_of_credit_card_clients.data.targets

# Credit Card Fraud Detection (Kaggle) veri setini yükle
df_fraud = pd.read_csv('creditcard.csv')

# Veri setlerinin ilk 5 satırını göster
print('Credit Card Default Prediction (UCI) - X_default.head():')
print(X_default.head())
print('\nCredit Card Default Prediction (UCI) - y_default.head():')
print(y_default.head())
print('\nCredit Card Fraud Detection (Kaggle) - df_fraud.head():')
print(df_fraud.head())

# Veri setlerinin istatistiksel özetlerini göster
print('\nCredit Card Default Prediction (UCI) - X_default.describe():')
print(X_default.describe())
print('\nCredit Card Default Prediction (UCI) - y_default.describe():')
print(y_default.describe())
print('\nCredit Card Fraud Detection (Kaggle) - df_fraud.describe():')
print(df_fraud.describe())

# Veri setlerinin bilgi özetlerini göster
print('\nCredit Card Default Prediction (UCI) - X_default.info():')
X_default.info()
print('\nCredit Card Default Prediction (UCI) - y_default.info():')
y_default.info()
print('\nCredit Card Fraud Detection (Kaggle) - df_fraud.info():')
df_fraud.info()


