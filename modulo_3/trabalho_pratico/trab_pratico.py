import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, mutual_info_score

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
diabetes_df = pd.read_csv('diabetes_numeric.csv')
bloodtransf_df = pd.read_csv('bloodtransf.csv')
wine_df = pd.read_csv('wine.csv')

print(diabetes_df.head())
print(bloodtransf_df.head())
print(wine_df.head())
# Verificar o número de features e instâncias
print("Diabetes Dataset:")
print(diabetes_df.shape)
print("\nBlood Transfusion Dataset:")
print(bloodtransf_df.shape)
print("\nWine Dataset:")
print(wine_df.shape)
print("Dados faltantes no Diabetes Dataset:")
print(diabetes_df.isnull().sum())
print("\nDados faltantes no Blood Transfusion Dataset:")
print(bloodtransf_df.isnull().sum())
print("\nDados faltantes no Wine Dataset:")
print(wine_df.isnull().sum())

# Diabetes Dataset
X_diabetes = diabetes_df.drop('c_peptide', axis=1)
y_diabetes = diabetes_df['c_peptide']
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_diabetes, y_diabetes, test_size=0.37, random_state=5762)

# Blood Transfusion Dataset
X_blood = bloodtransf_df.drop('Class', axis=1)
y_blood = bloodtransf_df['Class']
X_train_blood, X_test_blood, y_train_blood, y_test_blood = train_test_split(X_blood, y_blood, test_size=0.37, random_state=5762)

# Wine Dataset
X_wine = wine_df.drop('class', axis=1)
y_wine = wine_df['class']
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.37, random_state=5762)

# Treinar o modelo
model_diabetes = LinearRegression()
model_diabetes.fit(X_train_diabetes, y_train_diabetes)

# Fazer previsões
y_pred_diabetes = model_diabetes.predict(X_test_diabetes)

# Avaliar as métricas
r2 = r2_score(y_test_diabetes, y_pred_diabetes)
mae = mean_absolute_error(y_test_diabetes, y_pred_diabetes)
mse = mean_squared_error(y_test_diabetes, y_pred_diabetes)

print(f"R²: {r2}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")

# Treinar o modelo
model_blood = SVC(kernel='rbf', probability=True)
model_blood.fit(X_train_blood, y_train_blood)

# Fazer previsões
y_pred_blood = model_blood.predict(X_test_blood)
y_pred_proba_blood = model_blood.predict_proba(X_test_blood)[:, 1]

# Avaliar as métricas
accuracy = accuracy_score(y_test_blood, y_pred_blood)
precision = precision_score(y_test_blood, y_pred_blood)
recall = recall_score(y_test_blood, y_pred_blood)
f1 = f1_score(y_test_blood, y_pred_blood)
roc_auc = roc_auc_score(y_test_blood, y_pred_proba_blood)

print(f"Acurácia: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUROC: {roc_auc}")

# Identificar o número de clusters mais adequado (usando o método do cotovelo)
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=5762)
    kmeans.fit(X_train_wine)
    inertia.append(kmeans.inertia_)

# Plotar o método do cotovelo (opcional)
import matplotlib.pyplot as plt
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo')
plt.show()

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