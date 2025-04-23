import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# distancia
from scipy.spatial.distance import pdist, hamming, cosine


def euclidean_distance(x: np.array, y: np.array):
    return np.sqrt(np.sum((x - y) ** 2))


x = np.array([1, 2])
y = np.array([5, 5])
d = euclidean_distance(x, y)
print('a distancia euclidiana de x e y:', d, type(d))
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

# produto interno
x = np.array([1, 0, 0, 1, 1, 1]).reshape(-1, 1)
y = np.array([1, 1, 0, 0, 1, 0]).reshape(-1, 1)
# produto interno T -> transposto
print(x.T @ y)
# produto interno
print(np.sum(x * y))
# funcoes python
print(np.dot(x.T, y))
print(np.dot(x.ravel(), y.ravel()))


def cosine_similarity(x: np.array, y: np.array):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


x = np.array([1, 0, 0, 1, 1, 1]).reshape(-1, 1)
y = np.array([1, 1, 0, 0, 1, 0]).reshape(-1, 1)
c = cosine_similarity(x.ravel(), y.ravel())
print(c, type(c))
