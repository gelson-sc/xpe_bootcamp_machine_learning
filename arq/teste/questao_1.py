from keras.applications.vgg16 import VGG16

vgg16Model = VGG16(weights=None, classifier_activation='softmax', input_shape=(128, 220, 3), classes=2)
# Método 1: Verificar o número total de parâmetros treináveis
print(f"Total de parâmetros treináveis: {vgg16Model.count_params()}")
# Método 2: Listar as camadas e seus parâmetros treináveis
# for i, layer in enumerate(vgg16Model.layers):
#     print(f"Camada {i}: {layer.name}, Parâmetros treináveis: {layer.count_params()}")
# Método 3: Ver o resumo completo do modelo
vgg16Model.summary()
# Método 4: Listar especificamente os pesos treináveis
trainable_weights = vgg16Model.trainable_weights
print(f"Número de tensores de pesos treináveis: {len(trainable_weights)}")

print(f"Nome dos tensores de pesos treináveis: {trainable_weights}")
