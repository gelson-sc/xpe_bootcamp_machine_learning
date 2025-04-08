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
#df = df.head(100)
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
# print(Test_X_Tfidf)

'''
PERGUNTA 11
Para sabermos qual o vocabulário aprendido pelo Corpus, usamos usamos o seguinte trecho de código:
O que esse vocabulário representa e qual é o seu tipo?
'''
print(Tfidf_vect.vocabulary_)

# Naive Bayes
# Classificador - Algoritmo - NB
# ajuste o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use a função precision_score para obter a precisão
print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)

# SVM
# Classificador - Algoritmo - SVM
# ajusta o conjunto de dados de treinamento no classificador
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use a função precision_score para obter a precisão
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)

"""
Com base na documentação do Scikilearn e dos algoritmos Naive Bayes e SVM apresentados em nossas aulas,
codifique um classificador Random Forest
(consulte a documentação do Scikit-learn e tome como exemplo os classificadores Naive 
Bayes e SVM implementados no Notebook)
e responda as seguintes questões:
"""
# PERGUNTA 12
# Considerando os valores de (n_estimators = 10, random_state = 0) e o conjunto de treino e
# teste como 70/30, o Random Forest teve a sua acurácia prevista na faixa de qual porcentagem?
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_final'], df['label'], test_size=0.3)
forest = RandomForestClassifier(n_estimators=10, random_state=0)
forest.fit(Train_X_Tfidf, Train_Y)
predictions_rdf = forest.predict(Test_X_Tfidf)
print("Random Forest Accuracy 70/30 Score -> ", accuracy_score(predictions_rdf, Test_Y) * 100)

# PERGUNTA 13
# Considerando os valores de (n_estimators = 100, random_state = 0) e o conjunto de treino e teste como 80/20,
# o Random Forest, Naive Bayes e SVM, em relação a acurácia obtida, marque a alternativa correta...
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_final'], df['label'], test_size=0.2)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(Train_X_Tfidf, Train_Y)
predictions_rdf = forest.predict(Test_X_Tfidf)
print("Random Forest 80/20 Accuracy Score -> ", accuracy_score(predictions_rdf, Test_Y) * 100)
# PERGUNTA 14
# Considerando os valores de (n_estimators = 100, random_state = 0) e o conjunto de treino e teste como 80/20
# em relação ao Random Forest, a seguinte afirmação está correta...

# PERGUNTA 15
# Pensando na perspectiva de melhoria dos modelos de Machine Learning, podemos avaliar o ajuste de hiper parâmetros,
# considerando as seguintes técnicas...


# PARA SE PENSAR...
# Como saber se o nosso modelo criado está generalizando de maneira adequada?

# - A base possui um tamanho adequado?
# - O classificador é adequado para o problema em questão?


# Classificador - Algoritmo - RF
# Needed for the next step in model parameter tuning
# Train_X, Test_X, Train_Y, Test_Y

# random forest test
# Instantiate classifier
### SEU CODIGO AQUI ###

# fit on training data
### SEU CODIGO AQUI ###

# prever os rótulos no conjunto de dados de validação
### SEU CODIGO AQUI ###

# Use a função precision_score para obter a precisão
### SEU CODIGO AQUI ###

# Seeing the metrics
# print("Accuracy on training set: {:.3f}".format(forest.score(Train_X_Tfidf,Train_Y)))
# print("Accuracy on test set: {:.3f}".format(forest.score(Test_X_Tfidf, Test_Y)))
