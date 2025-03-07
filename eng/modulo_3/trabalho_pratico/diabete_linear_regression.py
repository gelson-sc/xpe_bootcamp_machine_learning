import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

diabetes_df = pd.read_csv('diabetes_numeric.csv')
print(diabetes_df.head())
print("Diabetes Dataset:")
print(diabetes_df.shape)
print("Dados faltantes no Diabetes Dataset:")
print(diabetes_df.isnull().sum())

# Diabetes Dataset
X_diabetes = diabetes_df.drop('c_peptide', axis=1)
y_diabetes = diabetes_df['c_peptide']
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_diabetes, y_diabetes, test_size=0.37, random_state=5762)

# Treinar o modelo
model_diabetes = LinearRegression()
model_diabetes.fit(X_train_diabetes, y_train_diabetes)

# Fazer previsões
y_pred_diabetes = model_diabetes.predict(X_test_diabetes)

# Avaliar as métricas
r2 = r2_score(y_test_diabetes, y_pred_diabetes)
mae = mean_absolute_error(y_test_diabetes, y_pred_diabetes)
mse = mean_squared_error(y_test_diabetes, y_pred_diabetes)

print(f"R²: {r2}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
# matriz_confusao = confusion_matrix(y_test_diabetes, y_pred_diabetes)
# print(matriz_confusao)
