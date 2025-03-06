import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
# https://www.kaggle.com/datasets/meet3010/cars-brand-prediction
df = pd.read_csv('cars.csv')
print(df.head(10))
# print(df)
df['cubicinches'] = pd.to_numeric(df['cubicinches'], errors='coerce')
df['weightlbs'] = pd.to_numeric(df['weightlbs'], errors='coerce')
print(df.describe())
print('duplicated', df.duplicated().sum())

print(df.shape)
print(df.isnull().sum())

print('Count cubicinches \n', df['cubicinches'].value_counts().sort_values(ascending=False))
print('Count year \n', df['year'].value_counts().sort_values(ascending=False))
print('Count brand \n', df['brand'].value_counts().sort_values(ascending=False))
# trocar brand para int
le = LabelEncoder()
df['brand_int'] = le.fit_transform(df['brand'])

# APAGAR linhas NA
df.dropna()
#df['cubicinches'].fillna(df['cubicinches'].mean(), inplace=True)
print(df.head(5))
print(df.info())
print(df.shape)
print(df.isna().sum())

X = df.drop(['brand'],axis=1)
y =  df['brand']
#clnd_data = df.dropna()
#data_df2.isnull().mean().sort_values(ascending=False) * 100 # We still see the missing value