import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, mutual_info_score, confusion_matrix

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

wine_df = pd.read_csv('wine.csv')

print(wine_df.head())

print("\nWine Dataset:")
print(wine_df.shape)
print("\nDados faltantes no Wine Dataset:")
print(wine_df.isnull().sum())

X_wine = wine_df.drop('class', axis=1)
y_wine = wine_df['class']
#print(y_wine)
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.37, random_state=5762)

# Identificar o número de clusters mais adequado (usando o método do cotovelo)
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=5762)
    kmeans.fit(X_train_wine)
    inertia.append(kmeans.inertia_)

# Plotar o método do cotovelo (opcional)
# import matplotlib.pyplot as plt
# plt.plot(range(2, 11), inertia, marker='o')
# plt.xlabel('Número de Clusters')
# plt.ylabel('Inércia')
# plt.title('Método do Cotovelo')
# plt.show()

# Escolher o número de clusters (por exemplo, 3)
kmeans = KMeans(n_clusters=3, random_state=5762)
kmeans.fit(X_train_wine)

# Fazer previsões
y_pred_wine = kmeans.predict(X_test_wine)

# Avaliar as métricas
silhouette = silhouette_score(X_test_wine, y_pred_wine)
davies_bouldin = davies_bouldin_score(X_test_wine, y_pred_wine)
mutual_info = mutual_info_score(y_test_wine, y_pred_wine)

print(f"Coeficiente de Silhueta: {silhouette}")
print(f"Davies-Bouldin Score: {davies_bouldin}")
print(f"Mutual Information: {mutual_info}")

matriz_confusao = confusion_matrix(y_test_wine, y_pred_wine)
print(matriz_confusao)
