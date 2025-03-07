import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregando o dataset
df = pd.read_csv('cars_validade.csv')

# Visualizando as primeiras linhas
print("Primeiras linhas do dataset:")
print(df.head())

# Informações sobre o dataset
print("\nInformações sobre o dataset:")
print(df.info())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())

# Verificando valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# Definindo as variáveis de entrada e saída
X = df[['cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60']]

# Definindo a classificação de eficiência com base no mpg (consumo)
# Vamos usar quartis para criar 3 classes de eficiência
q1 = df['mpg'].quantile(0.33)
q2 = df['mpg'].quantile(0.67)

def classify_efficiency(mpg):
    if mpg < q1:
        return 0  # Baixa eficiência
    elif mpg < q2:
        return 1  # Média eficiência
    else:
        return 2  # Alta eficiência

df['efficiency'] = df['mpg'].apply(classify_efficiency)
y = df['efficiency']

print("\nDistribuição das classes de eficiência:")
print(df['efficiency'].value_counts())

# Visualizando a relação entre as variáveis
plt.figure(figsize=(12, 10))
sns.heatmap(df[['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'efficiency']].corr(),
            annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# Parte 1: Análise de Componentes Principais (PCA)

# Normalizando os dados
normaliza = StandardScaler()
X_scaled = normaliza.fit_transform(X)

# Aplicando PCA - corrigido para 5 componentes (máximo possível)
pca = PCA(n_components=5)  # Corrigido para 5 componentes em vez de 7
X_pca = pca.fit_transform(X_scaled)

# Visualizando a variância explicada
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='g', label='Variância individual')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Variância acumulada')
plt.axhline(y=0.95, color='r', linestyle='--', label='Threshold 95%')
plt.xlabel('Componentes Principais')
plt.ylabel('Proporção de Variância Explicada')
plt.title('Variância Explicada por Componentes Principais')
plt.legend()
plt.show()

# Visualizando os loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=X.columns
)
print("\nLoadings dos componentes principais:")
print(loadings)

# Visualizando os dados nos primeiros dois componentes principais
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Classe de Eficiência')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Dados nos Primeiros Dois Componentes Principais')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Parte 2: Clusterização com K-Means

# Aplicando K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters

# Visualizando os clusters nos primeiros dois componentes principais
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')

# Transformando os centróides para o espaço PCA
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=100, label='Centróides')

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Clusters K-Means nos Primeiros Dois Componentes Principais')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# Comparando clusters com classes de eficiência
cluster_efficiency = pd.crosstab(df['cluster'], df['efficiency'])
print("\nComparação entre clusters e classes de eficiência:")
print(cluster_efficiency)

# Parte 3: Classificação com Algoritmos Supervisionados

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Normalizando os dados
X_train_scaled = normaliza.fit_transform(X_train)
X_test_scaled = normaliza.transform(X_test)

# Modelo de Árvore de Decisão
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_scaled, y_train)
y_pred_dt = dt_classifier.predict(X_test_scaled)

print("\nResultados da Árvore de Decisão:")
print("Acurácia:", accuracy_score(y_test, y_pred_dt))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_dt))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_dt))

# Modelo de Regressão Logística
lr_classifier = LogisticRegression(random_state=42, max_iter=1000)
lr_classifier.fit(X_train_scaled, y_train)
y_pred_lr = lr_classifier.predict(X_test_scaled)

print("\nResultados da Regressão Logística:")
print("Acurácia:", accuracy_score(y_test, y_pred_lr))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_lr))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_lr))

# Importância das features para a Árvore de Decisão
feature_importance = pd.DataFrame(
    {'Feature': X.columns, 'Importance': dt_classifier.feature_importances_}
).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Importância das Features na Árvore de Decisão')
plt.show()

# Comparando os modelos
models = ['Árvore de Decisão', 'Regressão Logística']
accuracies = [accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_lr)]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
plt.title('Comparação de Acurácia entre Modelos')
plt.ylim(0, 1)
plt.show()

# Parte 4: Análise adicional - Visualizando os clusters e classes de eficiência
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['cubicinches'], df['hp'], c=df['efficiency'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Classe de Eficiência')
plt.xlabel('Cilindradas Cúbicas')
plt.ylabel('Potência (hp)')
plt.title('Relação entre Cilindradas Cúbicas e Potência por Classe de Eficiência')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['weightlbs'], df['time-to-60'], c=df['efficiency'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Classe de Eficiência')
plt.xlabel('Peso (lbs)')
plt.ylabel('Tempo para 60 mph')
plt.title('Relação entre Peso e Tempo de Aceleração por Classe de Eficiência')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Estatísticas por cluster
cluster_stats = df.groupby('cluster').agg({
    'mpg': 'mean',
    'cylinders': 'mean',
    'cubicinches': 'mean',
    'hp': 'mean',
    'weightlbs': 'mean',
    'time-to-60': 'mean',
    'year': 'mean'
}).reset_index()

print("\nEstatísticas por cluster:")
print(cluster_stats)

# Estatísticas por classe de eficiência
efficiency_stats = df.groupby('efficiency').agg({
    'mpg': 'mean',
    'cylinders': 'mean',
    'cubicinches': 'mean',
    'hp': 'mean',
    'weightlbs': 'mean',
    'time-to-60': 'mean',
    'year': 'mean'
}).reset_index()

print("\nEstatísticas por classe de eficiência:")
print(efficiency_stats)

# Conclusão
num_components_for_95 = np.argmax(cumulative_variance >= 0.95) + 1
print("\nConclusão da análise:")
print(f"1. O PCA mostrou que {num_components_for_95} componentes principais explicam mais de 95% da variância nos dados.")
print("2. A clusterização K-Means identificou 3 grupos distintos de veículos.")
print("3. A Árvore de Decisão obteve uma acurácia de {:.2f}% na classificação de eficiência.".format(accuracy_score(y_test, y_pred_dt)*100))
print("4. A Regressão Logística obteve uma acurácia de {:.2f}% na classificação de eficiência.".format(accuracy_score(y_test, y_pred_lr)*100))
print(f"5. As características mais importantes para determinar a eficiência são {', '.join(feature_importance['Feature'].head(3).tolist())}.")