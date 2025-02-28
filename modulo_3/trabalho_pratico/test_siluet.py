import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

# Carregar dataset
df = pd.read_csv('wine.csv')

# Remover colunas não numéricas (se houver)
df = df.select_dtypes(include=[np.number])

# Separar dados em treino e teste
train_data, _ = train_test_split(df, test_size=0.37, random_state=5762)

# Testar diferentes valores de k
inertia = []
silhouette_scores = []
K_range = range(2, 8)  # Testando de 2 a 7 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=5762, n_init=10)
    labels = kmeans.fit_predict(train_data)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(train_data, labels))

# Melhor k baseado no Silhouette Score
best_k = K_range[np.argmax(silhouette_scores)]
best_silhouette = max(silhouette_scores)

# Retornar os resultados
print(best_k, best_silhouette)
