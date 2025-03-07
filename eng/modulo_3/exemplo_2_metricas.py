import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
print(df.head(10))
print(df.describe())
print(df.info())
print(df.shape)
print(df.isnull().sum())
# 🔹 2. Selecionar as features e a variável alvo
X = df[["rm"]]  # Número médio de quartos por residência
y = df["medv"]  # Preço médio das casas
scaler = MinMaxScaler()
X_scaled = X# scaler.fit_transform(X)
# 🔹 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# 🔹 4. Criar e treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# 🔹 5. Fazer previsões
y_pred = model.predict(X_test)

# 🔹 6. Calcular métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 🔹 7. Calcular R² ajustado
n = X_test.shape[0]  # Número de observações
p = X_train.shape[1]  # Número de variáveis independentes
r2_adjusted = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

# 🔹 8. Exibir resultados
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print(f"R² Ajustado: {r2_adjusted:.4f}")

#exit(0)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_test["rm"], y=y_test, label="Dados reais", alpha=0.7)
sns.lineplot(x=X_test["rm"], y=y_pred, color="red", linewidth=2, label="Regressão Linear")
plt.xlabel("Número médio de quartos (RM)")
plt.ylabel("Preço médio das casas (MEDV)")
plt.legend()
plt.title("Regressão Linear: RM vs MEDV")
plt.show()
