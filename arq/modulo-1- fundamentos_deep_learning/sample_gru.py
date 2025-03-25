import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Gerando uma série temporal simples (seno)
def create_time_series(n_samples=1000):
    t = np.linspace(0, 50, n_samples)
    x = np.sin(t) + np.random.normal(scale=0.1, size=n_samples)
    return x

# Criando a série temporal
data = create_time_series()
time_steps = 10  # Janela de tempo para entrada na GRU

# Criando os conjuntos de treinamento
X, y = [], []
for i in range(len(data) - time_steps):
    X.append(data[i:i+time_steps])
    y.append(data[i+time_steps])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Adicionando dimensão extra para entrada da GRU

# Criando o modelo GRU
model = tf.keras.Sequential([
    tf.keras.layers.GRU(50, activation='relu', return_sequences=True, input_shape=(time_steps, 1)),
    tf.keras.layers.GRU(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compilação e treinamento do modelo
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=16, verbose=1)

# Fazendo previsões
predictions = model.predict(X)

# Plotando os resultados
plt.plot(data, label="Real")
plt.plot(range(time_steps, len(predictions) + time_steps), predictions, label="Previsão", linestyle="dashed")
plt.legend()
plt.show()
