import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('cars_validade.csv')
print(df.head(10))
X = df.drop(['brand_id'],axis=1)
y =  df['brand_id']
# Normalizar os dados
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
