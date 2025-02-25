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
# ***************** Modelos Preditivos (Classificação) *****************
# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Padronizar os dados (opcional, mas recomendado para alguns modelos)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Treinar um modelo de SVM
model_svm = SVC(gamma='auto',kernel='rbf', random_state=1)
model_svm.fit(X_train, y_train)

# Previsões
y_pred_svm = model_svm.predict(X_test)

# Avaliação
print("\nSVM - Acurácia:", accuracy_score(y_test, y_pred_svm))
print("SVM - Relatório de Classificação:\n", classification_report(y_test, y_pred_svm))
cm_rf = confusion_matrix(y_test, y_pred_svm)
print("SVM - Matriz de Confusão:\n", cm_rf)


