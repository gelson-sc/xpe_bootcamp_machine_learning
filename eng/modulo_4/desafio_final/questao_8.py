import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('cars_validade.csv')
# Selecionar as colunas relevantes
selected_columns = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year']
df_selected = df[selected_columns]
# Padronizar os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)
# Aplicar PCA
pca = PCA()
pca.fit(df_scaled)
# Variância explicada pela primeira componente principal
explained_variance = pca.explained_variance_ratio_[0] * 100
print(f"A variância explicada pela primeira componente principal é: {explained_variance:.2f}%")