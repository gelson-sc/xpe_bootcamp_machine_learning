import pandas as pd
import numpy as np

ratings = np.array([
    [4.0, 0.0, 0.0, 4.7, 1.0, 0.0, 0.0],
    [5.0, 4.5, 4.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.5, 5.0, 4.0, 0.0],
    [4.1, 3.0, 0.0, 4.9, 0.0, 0.0, 3.0],
    [1.0, 4.0, 0.0, 2.5, 3.8, 1.0, 5.0],
])

ratings = pd.DataFrame(data=ratings,
                       columns=['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7'],
                       index=['U1', 'U2', 'U3', 'U4', 'U5'], dtype=float)

print(ratings)


def cosine_similarity(x: np.array, y: np.array):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def array_centering(v):
    '''subtrando dos elementos nao nulos pela media'''
    # copia para evitar sobrescrita
    v = v.copy()
    # idexacao para extrair elementos nao nulos
    non_zeros = v > 0
    # subsititucao pela media
    v[non_zeros] = v[non_zeros] - np.mean(v[non_zeros]) + 1e-6
    return v


def centered_cosine_similarity(x: np.array, y: np.array):
    # calcula a similaridade de cossenos centralizadda entre os arrays x e  y
    # centraliza os arrays
    x = array_centering(x)
    y = array_centering(y)
    # similaridade por cossenos
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


U1 = ratings.loc['U1'].values
U2 = ratings.loc['U2'].values
U3 = ratings.loc['U3'].values
U4 = ratings.loc['U4'].values
U5 = ratings.loc['U5'].values
print(f'U1: {U1}')
print(f'U2: {U2}')
print(f'U3: {U3}')
print(f'U4: {U4}')
print(f'U5: {U5}')
questao_a = cosine_similarity(U1, U2)
print(f'A:{questao_a:.2f}')
questao_b = cosine_similarity(U1, U3)
print(f'B:{questao_b:.2f}')
