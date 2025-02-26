import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Gerando dados sintéticos
X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

# Aplicando K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Cálculo do Coeficiente de Silhueta
silhouette_avg = silhouette_score(X, labels)
print(f"Coeficiente de Silhueta: {silhouette_avg:.2f}")

# Cálculo da Pureza
def purity_score(y_true, y_pred):
    from scipy.stats import mode
    clusters = np.unique(y_pred)
    majority_sum = 0
    for cluster in clusters:
        mask = y_pred == cluster
        majority_label = mode(y_true[mask])[0][0]
        majority_sum += np.sum(y_true[mask] == majority_label)
    return majority_sum / len(y_true)

purity = purity_score(y_true, labels)
print(f"Índice de Pureza: {purity:.2f}")