import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
df = pd.read_csv(url, header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
# Separar variáveis independentes (X) e dependente (y)
X = df.drop('class', axis=1)
y = df['class']
print('************** solve with K-Nearest Neighbors (KNN) *********************')
# Separar variáveis independentes (X) e dependente (y)
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf_KNN = KNeighborsClassifier(n_neighbors=5)
clf_KNN.fit(X_train, y_train)

y_pred_knn = clf_KNN.predict(X_test)

print("KNN - Acurácia:", accuracy_score(y_test, y_pred_knn))
print("KNN - Relatório de Classificação:\n", classification_report(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Falsa (0)', 'Autêntica (1)'],
            yticklabels=['Falsa (0)', 'Autêntica (1)'])
plt.title('Matriz de Confusão - KNN (k=5)')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()