import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.image import resize

# Carregar o dataset Fashion MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalizar os dados para o intervalo [0,1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Preparar os dados para a VGG-16
# Expandir a dimensão para adicionar um canal (de 28x28 para 28x28x1)
X_train_vgg = np.expand_dims(X_train, axis=-1)
X_test_vgg = np.expand_dims(X_test, axis=-1)

# Repetir o canal único três vezes para simular imagens RGB (de 28x28x1 para 28x28x3)
X_train_vgg = np.repeat(X_train_vgg, 3, axis=-1)
X_test_vgg = np.repeat(X_test_vgg, 3, axis=-1)

# Redimensionar as imagens para 32x32x3 para serem compatíveis com a VGG-16
X_train_vgg = resize(X_train_vgg, [32, 32])
X_test_vgg = resize(X_test_vgg, [32, 32])

# Construção da VGG-16
# Criamos a entrada da rede VGG-16
input_layer = keras.Input(shape=(32, 32, 3))

# Instanciamos a VGG-16 sem pesos pré-treinados (weights=None) e sem a camada de saída
vgg_base = VGG16(include_top=False, weights=None, input_tensor=input_layer)

# Adicionamos uma camada de Flatten e uma camada densa para classificação
x = layers.Flatten()(vgg_base.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(10, activation="softmax")(x)

# Criamos o modelo final
vgg_model = keras.Model(inputs=input_layer, outputs=x)
vgg_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# Construção da rede personalizada
def build_custom_model():
    model = keras.Sequential([
        # Camada de entrada: achatar a imagem de 28x28 para um vetor 784
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),

        # Primeira camada oculta: número de neurônios igual ao número de pixels (784)
        layers.Dense(28 * 28, activation="relu"),

        # Segunda camada oculta: 1024 neurônios
        layers.Dense(1024, activation="relu"),

        # Terceira camada oculta: 2048 neurônios
        layers.Dense(2048, activation="relu"),

        # Quarta camada oculta: 2048 neurônios
        layers.Dense(2048, activation="relu"),

        # Camada de saída: 10 neurônios (um para cada classe do Fashion MNIST) com softmax
        layers.Dense(10, activation="softmax")
    ])

    # Compilar o modelo com otimizador Adam e função de perda adequada para classificação
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# Criar a rede personalizada
custom_model = build_custom_model()

# Treinar os modelos
vgg_model.fit(X_train_vgg, y_train, epochs=10, validation_data=(X_test_vgg, y_test))
custom_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
