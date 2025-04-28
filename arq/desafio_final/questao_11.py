import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Carregar dataset
df = pd.read_csv('/home/gelson/datasets/creditcard_original.csv')

# 2. Normalizar 'Time' e 'Amount'
scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

# 3. Separar entrada e saída
X = df.drop('Class', axis=1)
y = df['Class']

# 4. Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Treinar MLP
#mlp = MLPClassifier(random_state=42, max_iter=300)  # número de iterações suficiente para convergir
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=(10,), activation='relu', solver='adam', random_state=1)
mlp.fit(X_train, y_train)

# 6. Avaliar
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f'Acurácia: {acc:.2f}')
