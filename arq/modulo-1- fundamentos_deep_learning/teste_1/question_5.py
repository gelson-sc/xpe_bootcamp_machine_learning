from keras.applications import VGG16
from keras.datasets import fashion_mnist
import numpy as np

# Carregar o MNIST original (28x28x1)
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_train.shape)
print(X_test.shape)

# Tentar usar a VGG16 padrão (224x224x3) sem redimensionar
model = VGG16(weights=None, input_shape=(224, 224, 3))  # Causará erro!
print(model.summary())
# Tentar usar uma VGG16 adaptada (32x32x3) sem redimensionar
model_adapted = VGG16(weights=None, input_shape=(32, 32, 3))  # Também causará erro se o input for 28x28x1