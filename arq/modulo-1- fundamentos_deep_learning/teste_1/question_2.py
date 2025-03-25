from keras.applications import VGG16

model = VGG16(weights=None)  # Sem pesos pré-treinados para contar apenas a arquitetura
total_params = model.count_params()
trainable_params = sum([w.numpy().size for w in model.trainable_weights])

print(f"Total de parâmetros: {total_params}")
print(f"Parâmetros treináveis: {trainable_params}")