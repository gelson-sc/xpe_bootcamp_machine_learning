import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv('winequality-red.csv', sep=';')

# Separar features e target
X = df.drop('quality', axis=1)
y = df['quality']

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizar as features (obrigatório para SVM)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar o modelo SVM
svm = SVC(kernel='rbf', random_state=42)  # kernel='rbf' é o padrão e funciona bem na maioria dos casos
svm.fit(X_train, y_train)

# Fazer previsões
y_pred = svm.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Gerar o relatório de classificação
# print("\nRelatório de Classificação:")
# print(classification_report(y_test, y_pred))

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(conf_matrix)

# Visualizar a matriz de confusão com Seaborn
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=svm.classes_, yticklabels=svm.classes_)
# plt.xlabel('Previsão')
# plt.ylabel('Real')
# plt.title('Matriz de Confusão - SVM')
# plt.show()