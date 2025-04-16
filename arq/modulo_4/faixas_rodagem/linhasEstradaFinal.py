import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt


def main():
    # 1 - obter a imagem colorida e transformar em escala de cinza
    img = cv2.imread('rodovia2.png')
    print("Dimensões da Imagem: ", img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # aplica a escala de cinza
    cv2.imwrite('roi_imag_gray.png', gray)  # cria a imagem em escala de cinza
    #show_image(gray)
    kernel_size = 5  # indica a quantidade de pixels a serem utilizados para
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)  # reduz o ruído da imagem
    cv2.imwrite('roi_imag_gray_blur.png', blur_gray)  # cria a imagem com a redução de ruído
    # 2 - Encontra as bordas da imagem
    low_threshold = 50
    high_threshold = 150
    # essa função é utilizada para realizar a detecção das "bordas" da imagem, ou seja
    # os pontos que determinam as fronteiras da estrutura dos objetos da imagem.
    img_bordas = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # compara diferentes parâmetros da função  Canny
    img_bordas2 = cv2.Canny(blur_gray, 60, 160)
    img_compara_parametros = np.hstack((gray, img_bordas, img_bordas2))
    # encontra o polígono que contém as retas na imagem (região de interesse)
    poligono_ROI = np.array([[0, 682], [0, 600], [500, 420], [530, 420], [1024, 600], [1024, 682]])

    # aplica a região de interesse
    img_roi = roi(img_bordas, poligono_ROI)
    cv2.imwrite('roi_imagem.png', img_roi)
    cv2.imwrite('testecanny.png', img_compara_parametros)  # cria a imagem de teste entre os diferentes
    # parâmetros para a i
    # 3 - Encontra as linhas utilizando o HoughTransform
    rho = 1  # menor distância entre cada pixel da imagem na identificação de uma linha
    theta = np.pi / 180  # resolução angular para definir uma linha (radianos)
    threshold = 50  # número mínimo de interseções para identificar uma linha
    min_line_length = 110  # Número mínimo de pixels que identifica uma linha
    max_line_gap = 30  # maior distância entre segmentos de pixels para formar uma linha
    line_image = np.copy(img) * 0  # vetor para a linha

    # função que aplica a transformação de Hough
    lines = cv2.HoughLinesP(img_roi, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # desenha o contorno das linhas encontradas sobre a imagem original
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)  # desenha sobre a imagem original

    cv2.imwrite('estrada_linhas.png', lines_edges)  # cria a imagem de saída


# função que define a região de interesse na imagem (ROI - Region of Interest)
def roi(img, vertices):
    # define a mascara para a imagem
    mask = np.zeros_like(img)
    # preenche a máscara para a figura
    cv2.fillPoly(mask, np.int32([vertices]), (255, 255, 255))
    # recorta a porção da imagem correspondente à mascara
    masked = cv2.bitwise_and(img, mask)
    return masked


def show_image(img):
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    print("iniciando")
    main()
