import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
import sys

lista_media = [5.86416667, 3.08833333, 3.79583333, 1.21916667]
lista_desvio = [0.8403169871476411, 0.41396121664823726, 1.7801870610197745, 0.7778491534710024]


def main():
    print('start models')
    slength =5.1
    swidth = 2.5
    plength = 3.0
    pwidth = 3
    lista_recebida = [float(slength), float(swidth), float(plength), float(pwidth)]
    lista_ajustada = ajustando_entradas(lista_recebida, lista_media, lista_desvio)
    print('lista_ajustada', lista_ajustada)
    classe = previsao_iris(lista_ajustada)
    if int(classe) == 0:
        previsao = 'Setosa'
    elif int(classe) == 1:
        previsao = 'Versicolor'
    else:
        previsao = "Virginica"
    print(previsao)


def ajustando_entradas(lista_entradas, lista_media, lista_desvio):
    # z= (x-u)/s
    lista_ajustada = []
    for (x, u, s) in zip(lista_entradas, lista_media, lista_desvio):
        z = (x - u) / s
        lista_ajustada.append(z)

    return lista_ajustada


def previsao_iris(lista_valores_formulario):
    prever = np.array(lista_valores_formulario).reshape(1, 4)  # transforma os valores do formulario
    modelo_salvo = joblib.load('modelo_knn_update.joblib')  # realiza a carga do modelo salvo
    resultado = modelo_salvo.predict(prever)  # aplica a previsao

    return resultado[0]


if __name__ == '__main__':
    main()
