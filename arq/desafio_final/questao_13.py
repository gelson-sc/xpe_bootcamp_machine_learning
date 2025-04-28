import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Carregar dataset
df = pd.read_csv('/home/gelson/datasets/creditcard_original.csv')

# 2. Normalizar 'Time' e 'Amount'
scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

# 3. Separar entradas e saídas
X = df.drop('Class', axis=1)
y = df['Class']

# 4. Subamostragem
fraudes = df[df['Class'] == 1]
normais = df[df['Class'] == 0].sample(n=len(fraudes), random_state=42)
df_balanceado = pd.concat([fraudes, normais])

# 5. Separar entrada e saída do balanceado
X_balanceado = df_balanceado.drop('Class', axis=1)
y_balanceado = df_balanceado['Class']

# 6. Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_balanceado, y_balanceado, test_size=0.3, random_state=42)

# 7. Treinar regressão logística
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 8. Avaliar
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f'Acurácia: {acc:.2f}')
