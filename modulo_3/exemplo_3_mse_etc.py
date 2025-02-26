import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# 🔹 2. Selecionar a feature e a variável alvo
X = df[["rm"]]  # Número médio de quartos por residência
y = df["medv"]  # Preço médio das casas

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 🔹 5. Fazer previsões
y_pred = model.predict(X_test)

# 🔹 6. Garantir que todas as previsões sejam maiores que zero (para RMSLE)
y_pred = np.maximum(y_pred, 1)  # Substitui valores negativos por 1

# 🔹 7. Calcular métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# RMSLE - Agora não teremos problemas com log de valores negativos
y_test_log = np.log1p(y_test)
y_pred_log = np.log1p(y_pred)
rmsle = np.sqrt(mean_squared_error(y_test_log, y_pred_log))

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"RMSLE: {rmsle:.4f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_test["rm"], y=y_test, label="Dados reais", alpha=0.7)
sns.lineplot(x=X_test["rm"], y=y_pred, color="red", linewidth=2, label="Regressão Linear")
plt.xlabel("Número médio de quartos (RM)")
plt.ylabel("Preço médio das casas (MEDV)")
plt.legend()
plt.title("Regressão Linear: RM vs MEDV")
plt.show()
