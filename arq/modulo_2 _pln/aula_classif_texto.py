import pandas as pd
import numpy as np
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

"""
PERGUNTA 1
Insira os modulos do NLTK para fazer download
"""
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
# %% md
## Definir sementes aleatórias

# Isso é usado para reproduzir o mesmo resultado todas as vezes se o script
# for mantido consistente, caso contrário, cada execução produzirá
# resultados diferentes. A semente pode ser definida para qualquer número.
np.random.seed(500)
df = pd.read_csv('corpus.csv', encoding='latin-1')
df = df.head(100)
"""
PERGUNTA 3
Qual o tipo de dados da variável Corpus criada?
"""
print(type(df))
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df['label'].value_counts())

# PERGUNTA 4 Para remover linhas em brancos se houver:
df['text'].dropna(inplace=True)

"""
PERGUNTA 5
Para passar todo o texto para letras minusculas, usamos o seguinte trecho de codigo:
"""
df['text'] = [entry.lower() for entry in df['text']]
print(df.head())

"""
PERGUNTA 6
Para quebrar o corpus em um conjunto de palavras, usamos o seguinte trecho de código:
"""
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
df['text'] = [word_tokenize(entry) for entry in df['text']]
print(df.head())
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
"""
PERGUNTA 7
Para fazermos o mapa de taggeamento das palavras em Adjetivo, Verbo e Adverbio, usamos o seguinte trecho de código:
"""
tag_map = defaultdict(lambda: wn.NOUN)
print(tag_map)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index, entry in enumerate(df['text']):
    Final_words = []
    # PERGUNTA 8 Para iniciar o WordNet lemmatizer, usamos o seguinte trecho de código:
    word_Lemmatized = WordNetLemmatizer()
    # post_tags
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    df.loc[index, 'text_final'] = str(Final_words)
print(df.head(10))
"""
PERGUNTA 9
Para separar o conjunto entre treino e teste com 70% para treino e 30% para teste, usamos o seguinte trecho de código:
"""
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_final'], df['label'], test_size=0.3)
"""
PERGUNTA 10
Para transformar dados categóricos do tipo string no conjunto de dados em valores numéricos que o modelo pode entender,
usamos o seguinte trecho de código:
"""
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
print(Train_Y)
print(Test_Y)

"""
PERGUNTA 10
Ao utilizar o TF-IDF, com o tamanho máximo do vocabulário definido em 5000, qual trecho de código devemos utilizar?
"""
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
#print(Test_X_Tfidf)

'''
PERGUNTA 11
Para sabermos qual o vocabulário aprendido pelo Corpus, usamos usamos o seguinte trecho de código:
O que esse vocabulário representa e qual é o seu tipo?
'''
print(Tfidf_vect.vocabulary_)

