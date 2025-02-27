import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_squared_log_error, accuracy_score)
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('diabetes_numeric.csv')
print(df.head(5))
print('Amostras e Features:', df.shape)
print(df.columns)
# print(df.describe())
# print(df.info())
# print(df.shape)
# print(df.isnull().sum())
feature_list = list(df.columns)

X = df.drop('c_peptide', axis=1)
y = df['c_peptide']
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.37, random_state=5762)
# scaler = StandardScaler()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print(y_pred)
# acuracia = accuracy_score(y_test, y_pred)
#
# print('ACURACIA:', acuracia)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmlse = mean_squared_log_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('R2:', r2)
print('MSE:', mse)
print('RMSE:', rmse)
print('RMSLE:', rmlse)
print('MAE:', mae)
