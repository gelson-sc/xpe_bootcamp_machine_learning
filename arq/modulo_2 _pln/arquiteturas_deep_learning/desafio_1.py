import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding  # empacotamento de palavras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
''' QUESTÃO 10 '''
model.add(Dense(1,activation='sigmoid'))
''' QUESTÃO 11 '''
#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
print(model.summary())
