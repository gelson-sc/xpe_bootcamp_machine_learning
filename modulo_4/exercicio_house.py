import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('kc_house_data.csv')
#print(df.head())
# verivicar dados
print('duplicated', df.duplicated().sum())
print(df.describe())
print('shape', df.shape)
# print('is_null \n', df.isnull().sum())
# print('is_na \n', df.isna().sum())
print('Count grades \n', df['grade'].value_counts())
print('Count condition \n', df['condition'].value_counts())
# converter datetime
df['date'] = pd.to_datetime(df['date'])
df['age'] = 2025 - df['yr_built']
df['renovation_age'] = df['yr_renovated'].apply(lambda x: 2025 - x if x > 0 else 0)
df = df.drop(['id', 'lat', 'long', 'yr_built', 'yr_renovated'], axis=1)
print(df.head())
# print(df.info())
#max
print(df[df['price']==df['price'].max()])
# correlacoes
print(df.corr(numeric_only=True)['price'].sort_values(ascending=False))

X = df.drop('price', axis=1)
y = df['price']
print(y.head())
print(X['bedrooms'].max())
print(y.max())



