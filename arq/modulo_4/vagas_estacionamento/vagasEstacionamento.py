import sys

import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt


def main():
    # 1 - obter a imagem colorida e transformar em escala de cinza
    # Esse processo de transformar uma imagem de RGB para a escala de cinza é importante,
    # pois reduzimos a quantidade de valores a serem analisados pelos algoritmos.
    # Ou seja, em RGB as cores de cada pixel é dada entre um intervalo de 0-255, quando
    # transformamos os valores para escala de cinza, podem ser utilizados valores entre 0-7, por exemplo.
    img = cv2.imread('estacionamento.jpg')
    # show_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show_image(gray)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # A função GaussianBlur é utilizada para "reduzir" a qualidade da imagem e
    # torná-la mais fácil de ser analisada pelos algoritmos de detecção.
    # por meio dessa função é possível reduzir o "ruído" na imagem
    #show_image(blur_gray)
    # 2 -
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    edges2 = cv2.Canny(blur_gray, 60, 120)
    images = np.hstack((gray, edges, edges2))
    # show_image(images)
    # essa função é utilizada para realizar a detecção das "bordas" da imagem, ou seja
    # os pontos que determinam as fronteiras da estrutura dos objetos da imagem.

    cv2.imwrite('testecanny.png', images)

    # 3 - Encontra as linhas utilizando o HoughTransform

    rho = 1  # menor distância entre cada pixel da imagem na identificação de uma linha
    theta = np.pi / 180  # resolução angular para definir uma linha
    threshold = 15  # número mínimo de interseções para identificar uma linha
    min_line_length = 50  # Número mínimo de pixels que identifica uma linha
    max_line_gap = 40  # maior distância entre segmentos de pixels para formar uma linha
    line_image = np.copy(img) * 0  # vetor para a linha

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    print(lines)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imwrite('vagas_identificadas_1.png', lines_edges)


def show_image(img):
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    print("iniciando")
    main()
