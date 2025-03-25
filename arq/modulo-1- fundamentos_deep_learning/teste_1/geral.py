import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess data for second network
x_train_flat = x_train.reshape(x_train.shape[0], 784).astype('float32') / 255
x_test_flat = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255
y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test, 10)

# Create second network
model = Sequential([
    Dense(784, activation='relu', input_shape=(784,)),  # Hidden 1
    Dense(1024, activation='relu'),                    # Hidden 2
    Dense(2048, activation='relu'),                    # Hidden 3
    Dense(2048, activation='relu'),                    # Hidden 4
    Dense(10, activation='softmax')                    # Output
])

# Model for VGG16
vgg_model = VGG16(weights=None, input_shape=(32, 32, 3), classes=10)

# Answers to the questions
print(f"1) Element at index 4000 belongs to class: {class_names[y_train[4000]]}")
print(f"2) VGG16 trainable parameters: {vgg_model.count_params()}")
print(f"3) Second network trainable parameters: {model.count_params()}")
print(f"4) Pixels after transformation for VGG-Net: 32 * 32 * 3 = 3072")
print(f"5) Error if using original images with VGG-Net: Image size mismatch (28x28x1 vs 32x32x3)")
print(f"7) Pixels for second network: 784")
print(f"8) Test set element at index 4000 belongs to class: {class_names[y_test[4000]]}")
print(f"13) Iterations per epoch with batch size 128: {len(x_train) // 128}")

# Special note about ReLU in output layer
# Create the network
model = Sequential([
    Dense(784, activation='relu', input_shape=(784,)),  # Hidden 1
    Dense(1024, activation='relu'),                    # Hidden 2
    Dense(2048, activation='relu'),                    # Hidden 3
    Dense(2048, activation='relu'),                    # Hidden 4
    Dense(10, activation='softmax')                    # Output
])

# Compile the model to initialize weights
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Calculate bias for each layer
total_bias = 0
for layer in model.layers:
    bias_count = 0
    if hasattr(layer, 'bias'):
        bias_count = layer.bias.shape[0]
        total_bias += bias_count
    print(f"Layer {layer.name}: {bias_count} biases")

print(f"\n10) Total number of biases: {total_bias}")
penultimate_layer = model.layers[-2]
last_layer = model.layers[-1]

# Calculate weights
weights = model.get_weights()[-2]
num_weights = weights.shape[0] * weights.shape[1]

print(f"Penultimate layer shape: {penultimate_layer}")#output_shape
print(f"Last layer shape: {last_layer}")#output_shape
print(f"Shape of weights connecting these layers: {weights.shape}")
print(f"Number of weights: {num_weights}")