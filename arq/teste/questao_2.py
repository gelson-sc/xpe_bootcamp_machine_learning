from keras.applications.vgg19 import VGG19

vgg19 = VGG19(input_shape=(128, 220, 3), classes=2, weights=None)
print(f"Total de parâmetros treináveis: {vgg19.count_params()}")
# for i, layer in enumerate(vgg19.layers):
#     print(f"Camada {i}: {layer.name}, Parâmetros treináveis: {layer.count_params()}")
vgg19.summary()
# Método 4: Listar especificamente os pesos treináveis
trainable_weights = vgg19.trainable_weights
print(f"Número de tensores de pesos treináveis: {len(trainable_weights)}")
