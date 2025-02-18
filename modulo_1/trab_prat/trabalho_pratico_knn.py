import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, r2_score

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('winequality-red.csv', sep=';')
print(df.head())
# print(data.describe())
# print(data.info())
# print(data.shape)
# print(data.isnull().sum())

# KNN RESOLVE
clf_KNN = KNeighborsClassifier(n_neighbors=5)
X = df.drop('quality', axis=1)  # Features
y = df['quality']  # Target
print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train[:2])
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')
score = r2_score(y_test, y_pred)
print('score', score)
# Gerar o relatório de classificação
# print("\nRelatório de Classificação:")
# print(classification_report(y_test, y_pred))
# Gerar a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão KNN:")
print(conf_matrix)

# Visualizar a matriz de confusão com Seaborn
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=knn.classes_, yticklabels=knn.classes_)
# plt.xlabel('Previsão')
# plt.ylabel('Real')
# plt.title('Matriz de Confusão')
# plt.show()
