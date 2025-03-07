import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
# https://www.kaggle.com/datasets/meet3010/cars-brand-prediction
df = pd.read_csv('cars.csv')
print(df.head(5))
df['cubicinches'] = pd.to_numeric(df['cubicinches'], errors='coerce')
df['weightlbs'] = pd.to_numeric(df['weightlbs'], errors='coerce')
media_cubicinches = df['cubicinches'].mean()
media_weightlbs = df['weightlbs'].mean()
df['cubicinches'] = df['cubicinches'].fillna(media_cubicinches)
df['weightlbs'] = df['weightlbs'].fillna(media_weightlbs)
print(df.head())
print(df.isna().sum())
print(df.describe())
#6
# selected_columns = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year']
# df_selected = df[selected_columns]
# # Calcular a mediana para a coluna 'mpg'
# median_mpg = df_selected['mpg'].median()
# print(f"A mediana para a característica 'mpg' é: {median_mpg}")

#7
print('Count time-to-60 \n', df['time-to-60'].value_counts().sort_values(ascending=False))
