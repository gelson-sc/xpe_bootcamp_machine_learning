import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU use
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
# 1. Carregando o dataset
df = pd.read_csv('fall_detection.csv')

# Visualizar primeiras linhas
print(df.head())

# 2. Gráficos exploratórios
plt.figure(figsize=(12, 4))
sns.scatterplot(data=df, x='TIME', y='HR', hue='ACTIVITY', palette='viridis')
plt.title('Scatterplot: HR vs TIME por atividade')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']])
plt.title('Boxplot das principais variáveis fisiológicas')
plt.show()

df[['SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']].hist(bins=20, figsize=(12, 6))
plt.suptitle('Histograma das variáveis')
plt.show()

# 3. Normalização com MinMaxScaler
features = ['SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']
scaler = MinMaxScaler()
X = scaler.fit_transform(df[features])

# 4. Preparando os dados para LSTM
# LSTM espera 3D: (samples, timesteps, features)
# Vamos transformar cada linha em uma "sequência de 1 timestep"
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Labels
y = to_categorical(df['ACTIVITY'])

# Separação treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modelo LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Avaliação
loss, accuracy = model.evaluate(X_test, y_test)
print(f'\n📊 Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Gráfico de desempenho
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.show()
