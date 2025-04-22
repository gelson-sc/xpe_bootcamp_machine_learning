import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carrega o dataset
df = pd.read_csv('fall_detection.csv')

# Seleciona as variáveis de entrada
cols = ['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']

# Aplica o StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)

# Verifica o menor valor da variável SL
print("Menor valor em SL após scaler:", df_scaled['SL'].min())