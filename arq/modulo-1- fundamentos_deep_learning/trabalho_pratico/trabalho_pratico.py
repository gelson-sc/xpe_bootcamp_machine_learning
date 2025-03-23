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
item = 400
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
#print(X_train.shape, y_train.shape)
first_image = X_test[item]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape(28,28)
plt.imshow(first_image, cmap='gray')
plt.show()
print(y_test[item])
vgg16 = VGG16(input_shape=(32, 32, 3), classes = 10, weights=None)
vgg16.summary()

# treino
print(X_train.shape)
X_train_1 = np.expand_dims(X_train, axis=-1)
print(X_train_1.shape)
X_train_1 = np.repeat(X_train_1, 3, axis=-1)
print(X_train_1.shape)
X_train_resize = resize(X_train_1, [32,32])
print(X_train_resize.shape)

# teste
print('----------- TESTE ---------------')
print(X_test.shape)
X_test_1 = np.expand_dims(X_test, axis=-1)
print(X_test_1.shape)
X_test_1 = np.repeat(X_test_1, 3, axis=-1)
print(X_test_1.shape)
X_test_resize = resize(X_test_1, [32,32])
print(X_test_resize.shape)
# convert
print('-------------- CONVERT --------------')
y_train_h = np_utils.to_categorical(y_train)
y_test_h = np_utils.to_categorical(y_test)
num_classes = y_test_h.shape[1]
print('y_train_h', y_train_h.shape, 'X_train_resize ',X_train_resize.shape)
print( 'classes ', num_classes)
num_pixels = X_train_resize.shape[1] * X_train_resize.shape[2]
print(num_pixels)
vgg16.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = vgg16.fit(X_train_resize/255, y_train_h, validation_data=(X_test_resize/255, y_test_h),
                      epochs=5, verbose=1, batch_size = 1000)

scores = vgg16.evaluate(X_test_resize, y_test_h, verbose=0)
print("%s: %.2f%%" % (vgg16.metrics_names[1], scores[1]*100))


