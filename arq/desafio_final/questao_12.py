import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
normais = df[df['Class'] == 0]

# Amostrar a mesma quantidade de normais que de fraudes
normais_sub = normais.sample(n=len(fraudes), random_state=42)

# Dataset balanceado
df_balanceado = pd.concat([fraudes, normais_sub])

# 5. Separar entrada e saída do dataset balanceado
X_balanceado = df_balanceado.drop('Class', axis=1)
y_balanceado = df_balanceado['Class']

# 6. Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_balanceado, y_balanceado, test_size=0.3, random_state=42)

# 7. Contar fraudes no treinamento
fraudes_treino = sum(y_train == 1)
print(f'Fraudes no treino: {fraudes_treino}')
