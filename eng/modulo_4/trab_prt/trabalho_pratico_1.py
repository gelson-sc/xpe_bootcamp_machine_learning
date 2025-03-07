from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
#import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#print(tf.__version__)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('kc_house_data.csv')
df['date'] = pd.to_datetime(df['date'])
# print(df.head())
# print('duplicated', df.duplicated().sum())
# print(df.describe())
# print('shape', df.shape)
# print('info', df.info())

# Calcular a correlação entre 'bathrooms' e 'sqft_living'
# correlation = df['bathrooms'].corr(df['sqft_living'])
# print("Correlação entre bathrooms e sqft_living:", correlation)

# Filtrar os imóveis com dois andares
two_floors_df = df[df['floors'] == 2]
# Contar a quantidade de imóveis com dois andares
num_two_floors = two_floors_df.shape[0]
# Exibir a quantidade
print(f"Quantidade de imóveis com dois andares: {num_two_floors}")
# Criar o histograma
plt.hist(df['floors'], bins=range(1, 5), edgecolor='black')
plt.xlabel('Número de Andares')
plt.ylabel('Quantidade de Imóveis')
plt.title('Histograma de Imóveis por Número de Andares')
plt.xticks(range(1, 5))
plt.show()

