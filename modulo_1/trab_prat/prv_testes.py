import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, r2_score

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('winequality-red.csv', sep=';')
print(df.head())
# desvio_padrao = df['fixed acidity'].std()
# print(f"O desvio padrão da coluna 'fixed acidity' é: {desvio_padrao}")
#
# mediana = df['residual sugar'].median()
# print(f"A mediana da coluna 'residual sugar' é: {mediana}")
#
# correlacao = df['fixed acidity'].corr(df['pH'], method='pearson')
# print(f"Coeficiente de correlação de Pearson ph: {correlacao}")
#
# correlacao = df['quality'].corr(df['alcohol'], method='pearson')
# print(f"Coeficiente de correlação de Pearson alcohol: {correlacao}")
#
# contagem = df['quality'].value_counts()
# # Acessando o número de instâncias com quality igual a 5
# instancias_quality_5 = contagem.get(5, 0)  # Retorna 0 se não houver quality igual a 5
# print(f"Número de instâncias com quality igual a 5: {instancias_quality_5}")
# contagem_quality = df['quality'].value_counts()
# print(contagem_quality)

# fixed_acidity = df[['fixed acidity']]
# scaler = MinMaxScaler()  # Criando o scaler com valores padrão (intervalo [0, 1])
# fixed_acidity_scaled = scaler.fit_transform(fixed_acidity)
# menor_valor_scaled = fixed_acidity_scaled.min()
# print(f"Menor valor da variável 'fixed acidity' após normalização: {menor_valor_scaled}")

# Suponha que o DataFrame seja df
df['quality_binary'] = df['quality'].apply(lambda x: 1 if x > 5 else 0)
# Features (X): todas as colunas, exceto 'quality' e 'quality_binary'
X = df.drop(['quality', 'quality_binary'], axis=1)

# Target (y): a coluna 'quality_binary'
y = df['quality_binary']
from sklearn.model_selection import train_test_split

# Dividindo os dados: 80% treino, 20% teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

# Criando o modelo
model = RandomForestClassifier(random_state=42)

# Treinando o modelo
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calculando a acurácia
acuracia = accuracy_score(y_test, y_pred)

print(f"Acurácia do modelo: {acuracia:.4f}")