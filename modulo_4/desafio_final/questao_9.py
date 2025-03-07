import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('cars_validade.csv')

selected_columns = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year']
df_selected = df[selected_columns]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

pca = PCA(n_components=3)
principal_components = pca.fit_transform(df_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(principal_components)
labels = kmeans.labels_
# Contar a quantidade de elementos em cada cluster
cluster_counts = pd.Series(labels).value_counts()
# Obter os centroides dos clusters
centroids = kmeans.cluster_centers_
print("Quantidade de elementos em cada cluster:")
print(cluster_counts)
print("\nCentroides dos clusters (utilizando as três componentes principais):")
print(centroids)