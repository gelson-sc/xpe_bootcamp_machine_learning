import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('cars_validade.csv')
# # Criar a coluna de eficiência do veículo
df['efficiency'] = df['mpg'] > 25
# Selecionar as colunas de entrada e saída
X = df[['cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60']]
y = df['efficiency']
# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Treinar o modelo de árvore de decisão
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
# Predizer a eficiência nos dados de teste
y_pred = clf.predict(X_test)
# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print('acuracia ', accuracy)