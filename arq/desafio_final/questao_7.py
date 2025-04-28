import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Carregar e preparar dados
df = pd.read_csv('/home/gelson/datasets/creditcard_original.csv')
scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

X = df.drop('Class', axis=1)
y = df['Class']

# 2. Aplicar KMeans com 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# 3. Criar um DataFrame para análise
df_clusters = pd.DataFrame({'Cluster': cluster_labels, 'Class': y})

# 4. Contar fraudes em cada cluster
fraude_por_cluster = df_clusters[df_clusters['Class'] == 1]['Cluster'].value_counts()
print(fraude_por_cluster)
