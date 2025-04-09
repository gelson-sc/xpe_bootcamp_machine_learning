import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding  # empacotamento de palavras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.layers import LSTM # forca CUDA CuDNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

nltk.download('stopwords')

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('train.csv')
print(df.head())
'''QUESTÃO 2'''
### Drop Nan Values
df = df.dropna()
## Get the Independent Features
X = df.drop('label', axis=1)
## Get the Dependent features
y = df['label']
print(X.shape, y.shape)
print(X.head())
print(y.head())
print(tf.__version__)
'''QUESTÃO 3'''
### Vocabulary size
voc_size = 5000  ### DEFINA O TAMANHO DO VOCABULÁRIO ###
# One-hot representation
messages = X.copy()
print(messages['title'][1:6])
messages.reset_index(inplace=True)

# '''QUESTÃO 4''''
### Stemming
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()  ### INSIRA AS LINHAS DE CODIGO PARA DIVIDIR UMA STRING EM UMA LISTA ###
    review = [ps.stem(word) for word in review if not word in set(
        stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

print(corpus[1:6])
# print(corpus)

# one hot representation
onehot_repr = [one_hot(words, voc_size) for words in corpus]
'''QUESTÃO 8'''
### IMPRIMA O CONTEUDO DO ONE HOT REPRESENTATION ###
print(onehot_repr[1:6])
'''QUESTÃO 9'''
sent_length = 20  ### PREENCHER COM O COMPRIMENTO MAXIMO DAS SEQUENCIAS ###
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
print(embedded_docs)
'''
FAÇA AS ALTERAÇÕES/TESTES CONFORME FOLHA DE PERGUNTAS 
DO DESAFIO PARA RESOLUÇÃO DAS QUESTÕES DE 10 A 15
'''
## Creating model
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model.add(LSTM(100))
''' QUESTÃO 10 '''
model.add(Dense(1, activation='sigmoid'))
''' QUESTÃO 11 '''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Adicionando o Dropout
embedding_vector_features = 40
model1 = Sequential()
model1.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model1.add(Bidirectional(LSTM(100)))
''' QUESTÃO 12 '''
model1.add(Dropout(0.5))
model1.add(Dense(1, activation='sigmoid'))
''' QUESTÃO 13 '''
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())
print(len(embedded_docs), y.shape)

X_final = np.array(embedded_docs)
y_final = np.array(y)
print(X_final.shape, y_final.shape)

'''QUESTÃO 14'''
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.30, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

### EXECUTE ESTE CÓDIGO CASO QUEIRA MODIFICAR O TAMANHO DO CONJUNTO DE TREINO E TESTE ###
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
'''

# Finally Training
''' QUESTÃO 15 AQUI '''
train_model = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
# train_model=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=64)
# train_model=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64)
exit(0)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# y_pred1=model.predict_classes(X_test)
y_pred1 = np.argmax(model.predict(X_test), axis=-1)
confusion_matrix(y_test, y_pred1)
accuracy_score(y_test, y_pred1)
print(classification_report(y_test, y_pred1))
plt.plot(train_model.history['accuracy'], 'b', label='train_accuracy')
plt.plot(train_model.history['val_accuracy'], 'r', label='val_accuracy')
plt.legend()
plt.show()
# pred_val=np.argmax(model.predict(x_val), axis=-1)
y_pred1 = np.argmax(model.predict(X_test), axis=-1)

cm = confusion_matrix(y_test, y_pred1)
plot_confusion_matrix(cm, figsize=(5, 5))
