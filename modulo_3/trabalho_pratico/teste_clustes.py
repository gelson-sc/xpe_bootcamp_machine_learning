import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

df = pd.read_csv("wine.csv")
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

# Plotando o Método do Cotovelo
# plt.figure(figsize=(10, 5))
# plt.plot(K_range, inertia, marker='o')
# plt.xlabel("Número de Clusters (k)")
# plt.ylabel("Inertia")
# plt.title("Método do Cotovelo")
# plt.show()

# Melhor k baseado no Silhouette Score
best_k = K_range[np.argmax(silhouette_scores)]
print(f"O número ideal de clusters segundo o coeficiente de Silhueta é: {best_k}")
