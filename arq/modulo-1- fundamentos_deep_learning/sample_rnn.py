import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.src.layers import SimpleRNN


#from tensorflow.keras import Sequential

# Gerando uma série temporal simples (seno)
def create_time_series(n_samples=1000):
    t = np.linspace(0, 50, n_samples)
    x = np.sin(t) + np.random.normal(scale=0.1, size=n_samples)
    return x

# Criando a série temporal
data = create_time_series()
time_steps = 10  # Janela de tempo para a RNN

# Criando os conjuntos de treinamento
X, y = [], []
for i in range(len(data) - time_steps):
    X.append(data[i:i+time_steps])
    y.append(data[i+time_steps])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Adicionando dimensão extra para entrada da RNN

# Criando o modelo RNN
model = Sequential([
    SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(time_steps, 1)),
    SimpleRNN(50, activation='relu'),
    Dense(1)
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
