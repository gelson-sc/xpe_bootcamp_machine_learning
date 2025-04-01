from keras.applications.resnet import ResNet152


resNet = ResNet152(input_shape=(128, 220, 3), classes=2, weights=None)
print(f"Total de parâmetros treináveis: {resNet.count_params()}")
# for i, layer in enumerate(resNet.layers):
#     print(f"Camada {i}: {layer.name}, Parâmetros treináveis: {layer.count_params()}")
resNet.summary()
# Método 4: Listar especificamente os pesos treináveis
trainable_weights = resNet.trainable_weights
print(f"Número de tensores de pesos treináveis: {len(trainable_weights)}")
