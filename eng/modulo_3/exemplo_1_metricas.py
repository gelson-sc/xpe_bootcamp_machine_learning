import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Gerando dados sintéticos
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Variável independente
y = 2.5 * X.squeeze() + np.random.randn(100) * 2  # Relação linear com ruído

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Calculando métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Calculando R² ajustado
n = X_test.shape[0]  # Número de observações
p = X_train.shape[1]  # Número de variáveis independentes
r2_adjusted = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

# Exibindo os resultados
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print(f"R² Ajustado: {r2_adjusted:.4f}")

# Plotando os resultados
plt.scatter(X_test, y_test, label="Dados reais")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regressão Linear")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
