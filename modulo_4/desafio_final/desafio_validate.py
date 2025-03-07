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
df['brand_id'] = df['brand'].map({' US.': 1, ' Japan.': 2, ' Europe.':3 })
df = df.drop(['brand'], axis=1)
print(df.head(5))
df['cubicinches'] = pd.to_numeric(df['cubicinches'], errors='coerce')
df['weightlbs'] = pd.to_numeric(df['weightlbs'], errors='coerce')
#print(df.describe())
#print('duplicated', df.duplicated().sum())
#print(df.shape)
#print(df.isnull().sum())
# print('Count cubicinches \n', df['cubicinches'].value_counts().sort_values(ascending=False))
# print('Count year \n', df['year'].value_counts().sort_values(ascending=False))
print('Count brand \n', df['brand_id'].value_counts().sort_values(ascending=False))
# trocar brand para int
#le = LabelEncoder()
#df['brand_int'] = le.fit_transform(df['brand'])

# APAGAR linhas NA
#df.dropna()
#df['cubicinches'].fillna(df['cubicinches'].mean(), inplace=True)


#df = df.dropna()
media_cubicinches = df['cubicinches'].mean()
media_weightlbs = df['weightlbs'].mean()
df['cubicinches'] = df['cubicinches'].fillna(media_cubicinches)
df['weightlbs'] = df['weightlbs'].fillna(media_weightlbs)
print(df.head())
print(df.isna().sum())
print(df.info())
print(df.shape)
# Correlacao
print(df.corr(numeric_only=True)['brand_id'].sort_values(ascending=False))
# Calcular a correlação entre 'bathrooms' e 'sqft_living'
correlation = df['cubicinches'].corr(df['cylinders'])
print("Correlação entre cubicinches e cylinders:", correlation)

# plt.figure(figsize=(12, 8))
# dfc = df.drop(['brand'], axis=1)
# sns.heatmap(dfc.corr(), annot=True, fmt='.2f', cmap='coolwarm')
# plt.title('Correlation Matrix of Features')
# plt.show()

X = df.drop(['brand_id'],axis=1)
y =  df['brand_id']

df.to_csv('cars_validade.csv', index=False)
df_c = pd.read_csv('cars_validade.csv')
print(df_c.head(10))
# sns.pairplot(df, hue='brand_id')
# plt.show()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Aplicar PCA
pca = PCA(n_components=2)  # Reduzir para 2 dimensões
X_pca = pca.fit_transform(X_scaled)
print(X_pca)
# Ver a variância explicada por cada componente
print("Variância explicada:", pca.explained_variance_ratio_)
print("Variância acumulada:", np.sum(pca.explained_variance_ratio_))
# Converter para DataFrame para melhor visualização
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
print(df_pca.head())