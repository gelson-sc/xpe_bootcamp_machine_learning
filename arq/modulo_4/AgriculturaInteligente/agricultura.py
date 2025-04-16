import os
import numpy as np
import tensorflow as tf
import keras
# from keras import backend as k #utiliza
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.image as mpimg
from mlxtend.plotting import plot_confusion_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU use
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))

caminho_treinamento = 'BD_treinamento'  # dividido em 2 pastas (sadias e contaminadas) - 60/60
caminho_validacao = 'BD_validacao'  # dividido em 2 pastas (sadias e contaminadas) - 20/20
caminho_teste = 'BD_teste'  # dividido em 2 pastas (sadias e contaminadas) - 25/25
# ---------------------------------------------------
# Conhecendo e Preparando o BD
# ----------------------------------------------------
# cria a batelada utilizando dados que estão no disco
# ImageDataGenerator - utilizada para adicionar as imagens e converter em um formato padrão (224x224)
batelada_treino = ImageDataGenerator().flow_from_directory(caminho_treinamento, target_size=(224, 224),
                                                           classes=['sadias', 'contaminadas'], batch_size=10)
batelada_validacao = ImageDataGenerator().flow_from_directory(caminho_validacao, target_size=(224, 224),
                                                              classes=['sadias', 'contaminadas'], batch_size=5)
batelada_teste = ImageDataGenerator().flow_from_directory(caminho_teste, target_size=(224, 224),
                                                          classes=['sadias', 'contaminadas'], batch_size=10)

# utilizado para interar sobre a batelada de dados
img, labels = next(batelada_treino)
# plt.figure()
# plt.imshow(img[0].astype(np.uint8))
# plt.title("{}".format(labels[0]))
# plt.show()
# criando o modelo de de classificação com rede convolucionária
model = Sequential()
# 32= número de neurônios na camada/ (3,3)= filtro utilizado para percorrer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
# utilizada para criar um vetor para a entrada de dados na camada de saída
model.add(Flatten())
# camada de saída da rede 2 neurônios. 10= sadia /01= contaminada
model.add(Dense(2, activation='softmax'))
model.summary()
# definindo o otimizador e a função perda
model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# treinamento do modelo
history = model.fit(
    batelada_treino,
    steps_per_epoch=12,
    validation_data=batelada_validacao,
    validation_steps=4,
    epochs=20
)
exit(0)
# deve ser utilizada, pois estamos realizando o treinamento via batelada
# steps_per_epoch = define a quantidade de epocas utilizadas para treinamento, baseando-se no numero de dados utilizados
# vamos utilizar 120 imagens para treinamento (60 sadias e 60 contaminadas), como a batelada é de 10, temos 120/10 = 12 vezes
# validation_data = utilizado para gerar a validação (compara o desempenho do treinamento com o valor real): a cada epoca de treinamento,
# compara o resultado obtido com a previsão realizada nas
# imagens de validação
# verbose=2 - indica o que desejamos exibir na saída do treinamento


# Lista os dados históricos do treinamento
print(history.history.keys())
# summarize history para a accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Treinamento', 'Teste'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Treinamento', 'Teste'], loc='upper left')
plt.show()

# ---------------------------------------------------
# Realiza a previsão do modelo
# ----------------------------------------------------

# previsão sobre qual imagem corresponde a cada elemento
teste_img, teste_labels = next(batelada_teste)

# utilizado para mostrar as imagens
plt.figure()
plt.imshow(teste_img[0].astype(np.uint8))
plt.title("{}".format(teste_labels[0]))
plt.show()  # mostra a imagem

# testar a classificação da imagens
teste_labels = teste_labels[:, 0]  # transforma sadias (10) em 1 e contaminadas (01) em apenas 0

# realiza a previsão utilizando os dados de teste
previsao = model.predict_generator(batelada_teste, steps=1, verbose=0)
# como no fit, devemos utilizar o generator, pois estamos utilizando as bateladas de dados
print(previsao)

# criando a matriz de confusão para comparar os resultados
matriz_confusao = confusion_matrix(teste_labels, previsao[:, 0])
nomes_das_classes = ['contaminadas', 'sadias']
fig, ax = plot_confusion_matrix(conf_mat=matriz_confusao,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=True,
                                class_names=nomes_das_classes)
plt.show()

# -----------------------------------------------------------------------------
#  Melhorando a prevesão do modelo - TRANSFER LEARNING
# ----------------------------------------------------------------------------


vgg16_model = tf.keras.applications.vgg16.VGG16()  # classe já pre-treinada para ser utilizada em nosso classificador

vgg16_model.summary()  # vamos ver como o modelo do vgg16 foi construído

print(type(vgg16_model))

# transformando o tipo model do vgg16 em sequencial
model = Sequential()  # cria um modelo sequencial
for layer in vgg16_model.layers[:-1]:  # extrai cada uma das camadas do vgg16
    model.add(layer)  # adiciona no modelo criado até a penultima camada

model.summary()
print(type(model))

# retirar a ultima camada do modelo, pois só desejamos classificar entre 2 grupos de imagens
# model.layers.pop()

# colocando as camadas intermediárias em modo de "hibernação"
for layer in model.layers:
    layer.trainable = False
# colocar em modo de hibernação, garante que, durante o treinamento, os pesos não serão atualizados

# adicionando a ultima camada para a classificação entre 2 grupos de imagens (cachorros ou gatos)
model.add(Dense(2, activation='softmax'))

# mostra o novo modelo CNN (nosso+vgg16)
model.summary()

# ----------------------------------------------------------------------------
#  Inicia o treinamento através dos novos pesos
# ---------------------------------------------------------------------------

# definindo o otimizador e a função perda
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# treinamento do modelo
history = model.fit_generator(batelada_treino, steps_per_epoch=12, validation_data=batelada_validacao,
                              validation_steps=4, epochs=20, verbose=2)

# previsão utilizando o modelo+VGG16

# previsão sobre qual imagem corresponde a cada elemento
teste_img, teste_labels = next(batelada_teste)

# utilizado para mostrar as imagens
plt.figure()
plt.imshow(teste_img[4].astype(np.uint8))  # seleciona a imagem da posição 4
plt.title("{}".format(teste_labels[4]))
plt.show()  # mostra a imagem

# testar a classificação da imagens
teste_labels = teste_labels[:, 0]  # transforma sadias 10 em 1 e contaminadas 01 em apenas 0

# realiza a previsão utilizando os dados de teste
previsao = model.predict_generator(batelada_teste, steps=1, verbose=0)
# como no fit, devemos utilizar o generator, pois estamos utilizando as bateladas de dados
print(previsao)

# criando a matriz de confusão para comparar os resultados
matriz_confusao = confusion_matrix(teste_labels, np.round(
    previsao[:, 0]))  # a diferença é que a rede gera valores float, então devemos converter
# em valores inteiros (0,1)
nomes_das_classes = ['contaminadas', 'sadias']
fig, ax = plot_confusion_matrix(conf_mat=matriz_confusao,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=True,
                                class_names=nomes_das_classes)
plt.show()

# Lista os dados históricos do treinamento
print(history.history.keys())
# summarize history para a accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Treinamento', 'Teste'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Treinamento', 'Teste'], loc='upper left')
plt.show()
