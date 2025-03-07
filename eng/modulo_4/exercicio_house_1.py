from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print(tf.__version__)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('kc_house_data.csv')
df['date'] = pd.to_datetime(df['date'])
# df.hist(bins=20, figsize=(20,20), color='g')
# plt.show()
#sns.pairplot(df)
# sns.scatterplot(x='sqft_living', y='price', data=df)
# plt.show()
# f, ax = plt.subplots(figsize=(20, 20))
# sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
# plt.show()
# maior correlacao com price é metragem 7.0 (sqft_living)
select_feacture = 'sqft_living'
X = df[select_feacture]
y = df['price']
print(X.head())
# Y target ou label que queremos determiar como resultado
print(y.head())

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.values.reshape(-1, 1))
print(X_scaled[1:5])
print(X_scaled.shape)
print(scaler.data_max_)
print(scaler.data_min_)
y = y.values.reshape(-1, 1)
print(y.shape)
y_scaled = scaler.fit_transform(y)
print(y_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25)
print(X_train.shape)
print('------------- CONSTRUINDO O REDE NEURAL ----------------')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu',input_shape=(1,)))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
print(model.summary())
model.compile(optimizer='adam', loss='mean_squared_error')
epochs_hist = model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=1, validation_split=0.2)
print(epochs_hist.history.keys())
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()
print(X_test.shape)
y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, "^", color='r')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values Vs Predictions')
plt.show()

y_predict_origin = scaler.inverse_transform(y_predict)
y_test_origin = scaler.inverse_transform(y_test)

plt.plot(y_test_origin, y_predict_origin, "^", color='r')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
k = X_test.shape[0:1]
k = len(X_test)
n = len(X_test)
print('k', k)
print('n', n)
mae = mean_absolute_error(y_test_origin, y_predict_origin)
mse = mean_squared_error(y_test_origin, y_predict_origin)
rmse = sqrt(mse)
r2 = r2_score(y_test_origin, y_predict_origin)
print('MAE',mae)
print('MSE',mse)
print('RMSE',rmse)
print('R2',r2)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)
print(adj_r2)
