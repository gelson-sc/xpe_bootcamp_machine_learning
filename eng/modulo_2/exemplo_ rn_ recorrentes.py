import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.losses import mean_squared_error

os.environ["KERAS_BACKEND"] = "jax"
import keras
from sklearn.preprocessing import MinMaxScaler

# print(keras.__version__)
df = pd.read_csv("AirPassengers.csv", index_col='Month')
print(df.head(10))
# plt_fg = df.plot(figsize=(10, 5))
# # mostra que não é uma série estacionaria
# plt.show()
series = df.values
print(series[:10])
# normalização (max 1 e min 0)
scaler = MinMaxScaler()# divisão em treino e teste
series = scaler.fit_transform(series)

# divisão em treino e teste
n, p = len(df), 0.7
train = series[:int(n*p)]
test = series[int(n*p):]

print('TRAIN', train[:10])
print('TESTE', test[:10])
# convertendo a série para uma estrutura x_train, y_train
x_train, y_train = train[:-1], train[1:]
x_test, y_test = test[:-1], test[1:]
# x_train, y_train: primeiras 5 linhas
print(np.c_[x_train, y_train][:5, :])

# construindo modelo
inputs = keras.layers.Input(shape=(1, 1))
lstm_out = keras.layers.LSTM(4)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()
# fit no modelo
model.fit(x=x_train, y=y_train, batch_size=1, epochs=25)
# previsão
forecast = model.predict(x_test)
print('forecast', forecast)
# valores na escala original
forecast_orig = scaler.inverse_transform(forecast)
y_train_orig = scaler.inverse_transform(y_train)
y_test_orig = scaler.inverse_transform(y_test)
# plot da predição
n_train, n_test = len(y_train), len(y_test)
plt.figure(figsize=(10, 5))
plt.plot(np.arange(n_train), y_train_orig,'.-', label='Treino')
plt.plot(np.arange(n_train, n_train+n_test), y_test_orig,'.-', label='Teste')
plt.plot(np.arange(n_train, n_train+n_test), forecast_orig,'.-', label='Predição')
plt.grid()
plt.legend()
plt.show()
# RMSE
rmse = np.sqrt(mean_squared_error(y_test_orig,forecast_orig))
print(f"RMSE do modelo no conjunto de teste: {rmse}")

# adição da sazonalidade como variável exógena
# [s_{t-12}, s_{t-1}] ~ s_t
sz = 12
x_train, y_train = np.c_[train[:-sz], train[sz-1:-1]], train[sz:]
x_test, y_test = np.c_[test[:-sz], test[sz-1:-1]], test

# contabilização do primeiro ciclo de teste na predição
x_test = np.r_[np.c_[train[-sz:],
                     np.r_[train[-1], test[:sz-1].ravel()]], x_test]

# construindo modelo
inputs = keras.layers.Input(shape=(2, 1))
lstm_out = keras.layers.LSTM(4)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

# fit no modelo
model.fit(x=x_train, y=y_train, batch_size=1, epochs=25)

# previsão
forecast = model.predict(x_test)

# valores na escala original
forecast_orig = scaler.inverse_transform(forecast)
y_train_orig = scaler.inverse_transform(y_train)
y_test_orig = scaler.inverse_transform(y_test)

# plot da predição
n_train, n_test = len(y_train), len(y_test)
plt.figure(figsize=(10, 5))
plt.plot(np.arange(n_train), y_train_orig,'.-', label='Treino')
plt.plot(np.arange(n_train, n_train+n_test), y_test_orig,'.-', label='Teste')
plt.plot(np.arange(n_train, n_train+n_test), forecast_orig,'.-', label='Predição')
plt.grid()
plt.legend()
plt.show()

# RMSE
rmse = np.sqrt(mean_squared_error(y_test_orig,forecast_orig))
print(f"RMSE do modelo no conjunto de teste: {rmse}")