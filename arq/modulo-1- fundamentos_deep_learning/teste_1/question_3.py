from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(784, activation='relu', input_shape=(784,)),  # Oculta 1
    Dense(1024, activation='relu'),                    # Oculta 2
    Dense(2048, activation='relu'),                    # Oculta 3
    Dense(2048, activation='relu'),                    # Oculta 4
    Dense(10, activation='softmax')                    # Saída
])

print(model.summary())  # Mostrará o total de parâmetros: 7.737.370