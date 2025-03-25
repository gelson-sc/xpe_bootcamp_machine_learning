from keras.datasets import fashion_mnist
import numpy as np
from matplotlib import pyplot as plt
# https://www.tensorflow.org/tutorials/keras/classification?hl=pt-br

# Carregar o dataset Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Definir os nomes das classes de acordo com a documentação do Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Obter o rótulo do elemento de índice 4000
label = y_train[4000]
print('label ', label)
# Mapear o rótulo para o nome da classe
class_name = class_names[label]
print(f"O elemento de índice 4000 pertence à classe: {class_name}")

# Obter a imagem e o rótulo do índice 4000
image_4000 = x_train[4000]
label_4000 = y_train[4000]

# Mostrar a imagem
plt.figure(figsize=(5, 5))
plt.imshow(image_4000, cmap='gray')
plt.title(f'Classe: {class_names[label_4000]} (Índice: {label_4000})')
plt.axis('off')  # Remove os eixos
plt.show()