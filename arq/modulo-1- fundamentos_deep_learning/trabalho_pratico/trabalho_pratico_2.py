from matplotlib import pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import utils as np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
import numpy as np
from tensorflow.image import resize
#Criar nova rede
(X_train_rede, y_train_rede), (X_test_rede, y_test_rede) = fashion_mnist.load_data()
num_pixels = X_train_rede.shape[1] * X_train_rede.shape[2]
print(X_train_rede.shape)
X_train2 = X_train_rede.reshape(X_train_rede.shape[0], num_pixels)
print(X_train2[0][333])
print(X_train2.shape)
X_test2 = X_test_rede.reshape(y_test_rede.shape[0], num_pixels).astype('float32')
y_train_rede_h = np_utils.to_categorical(y_train_rede)
y_test_rede_h = np_utils.to_categorical(y_test_rede)
X_train2 = X_train2/255
X_test2 = X_test2/255
model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(128, kernel_initializer='normal', activation = 'relu'))
model.add(Dense(10, kernel_initializer='normal', activation = 'softmax'))
model.summary()
