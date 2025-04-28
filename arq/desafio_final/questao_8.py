import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Carregar o dataset
df = pd.read_csv('/home/gelson/datasets/creditcard_original.csv')

# 2. Normalizar 'Time' e 'Amount'
scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

# 3. Separar entrada e saída
X = df.drop('Class', axis=1)
y = df['Class']

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Contar fraudes no teste
fraudes_no_teste = sum(y_test == 1)
print(fraudes_no_teste)