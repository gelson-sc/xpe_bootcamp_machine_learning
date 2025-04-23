# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# distancia
from scipy.spatial.distance import pdist, hamming, cosine
from sklearn.feature_extraction.text import CountVectorizer

"""Distancia euclidiana entre os vetores $n$-dimensionais$ $x$ e $y$:
d = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
"""


def euclidean_distance(x: np.array, y: np.array):
    return np.sqrt(np.sum((x - y) ** 2))


x = np.array([1, 2])
y = np.array([5, 5])
euclidean_distance(x, y)

# calculo normas
nrm = np.linalg.norm(x - y)
print(nrm, type(nrm))


def hamming_distance(x: np.array, y: np.array, normalize=False):
    factor = 1
    if normalize:
        factor = 1 / len(x)
    distance = factor * np.sum(np.abs(x != y))
    return distance


x = np.array([1, 0, 0, 1, 1, 0])
y = np.array([1, 1, 0, 0, 0, 0])
h = hamming_distance(x, y)
print('distancia de hamming:', h, type(h))
hn = hamming_distance(x, y, True)
print('distancia de hamming normalizada:', hn, type(hn))
# funcao pacote scipy
hnn = hamming(x, y)
print('distancia de hamming normalizada f:', hnn, type(hnn))

x = np.array([1, 0, 0, 1, 1, 1]).reshape(-1, 1)
y = np.array([1, 1, 0, 0, 1, 0]).reshape(-1, 1)
h = hamming_distance(x, y)
print('distancia de hamming:', h, type(h))

print(x)
print(y)

# produto interno T -> transposto
x.T @ y

# produto interno
np.sum(x * y)

# funcoes python
np.dot(x.T, y)

np.dot(x.ravel(), y.ravel())


def cosine_similarity(x: np.array, y: np.array):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


# calcula a similaridade de cossenos
x = np.array([1, 0, 0, 1, 1, 1]).reshape(-1, 1)
y = np.array([1, 1, 0, 0, 1, 0]).reshape(-1, 1)
c = cosine_similarity(x.ravel(), y.ravel())
print('similaridade de cossenos:', c)

# funcoies python
# cosine?

1 - cosine(x.ravel(), y.ravel())

"""# distancia euclidiana vs similaridade por cossenos"""

import wikipedia

wikipedia.set_lang("pt")

query_1 = wikipedia.page("Inteligência artificial")
query_2 = wikipedia.page("Futebol")
query_3 = wikipedia.page("Aprendizado de maquinas")
query_4 = wikipedia.page("Voleibol")

query_1.title

query_1.url

query_1.content

cv = CountVectorizer()
X = np.array(cv.fit_transform([query_1.content, query_2.content, query_3.content, query_4.content]).todense())

X.shape

X

# numero de palavras em cada pagina
print("Inteligencia Artificial", len(query_1.content.split()))
print("Futebol", len(query_2.content.split()))
print("Aprendizado de maquinas", len(query_3.content.split()))
print("Voleibol", len(query_4.content.split()))

# distancia euclidiana
print("Inteligencia Artificial x Futebol", euclidean_distance(X[0], X[1]))
print("Inteligencia Artificial x Aprendizado de maquinas", euclidean_distance(X[0], X[2]))
print("Inteligencia Artificial x Voleibol", euclidean_distance(X[0], X[3]))

# similaridade por cosenos
print("Inteligencia Artificial x Futebol", cosine_similarity(X[0], X[1]))
print("Inteligencia Artificial x Aprendizado de maquinas", cosine_similarity(X[0], X[2]))
print("Inteligencia Artificial x Voleibol", cosine_similarity(X[0], X[3]))

# categorizando um tweet
tweet = 'romário e ronaldo são os melhores atacantes que já vi jogar... dentro da pequena área era sempre gol!'
t = np.array(cv.transform([tweet]).todense())[0]

t

# tweets por euclidiana
print("tweet - IA", euclidean_distance(t, X[0]))
print("tweet - Futebol", euclidean_distance(t, X[1]))
print("tweet - Aprendizado de maquinas", euclidean_distance(t, X[2]))
print("tweet - Voleibol", euclidean_distance(t, X[3]))

# tweets por similaridade por cosseno
print("tweet - IA", cosine_similarity(t, X[0]))
print("tweet - Futebol", cosine_similarity(t, X[1]))
print("tweet - Aprendizado de maquinas", cosine_similarity(t, X[2]))
print("tweet - Voleibol", cosine_similarity(t, X[3]))

"""# o mau da dimesionalidade"""

n = 1000
user_a = np.ones(n)
user_b = np.zeros(n)
euclidean_distance = np.sqrt(np.cumsum((user_a - user_b) ** 2))
print(euclidean_distance)

# plot
plt.figure(figsize=(10, 5))
plt.plot(euclidean_distance, '.', label='Distancia euclidiana')
plt.xlabel('Dimesao $n$')
plt.ylabel('Distância')
plt.grid()
plt.show()
