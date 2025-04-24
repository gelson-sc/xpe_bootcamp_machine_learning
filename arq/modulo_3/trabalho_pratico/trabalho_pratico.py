import numpy as np


def main():
    # # questao 1
    # x = np.array([0, 1, 4])
    # y = np.array([5, 3, -1])
    # print('1', euclidean_distance(x, y))
    #
    # # questao 2
    # x2 = np.array([0, 1, 4])
    # y2 = np.array([0, 1, 4])
    # print('2', euclidean_distance(x2, y2))
    # questao 4
    # x = np.array([1, 0, 0, 1, 0, 0])
    # y = np.array([1, 1, 0, 0, 0, 0])
    # print('4.', hamming_distance(x, y))
    # print('5.', hamming_distance(x, y, True))

    x = np.array([1, 0])
    y = np.array([0, 1])
    print('6.', cosine_similarity(x, y))

    x = np.array([1, 0])
    y = np.array([2, 0])
    print('7.', cosine_similarity(x, y))

    x = np.array([1, 0])
    y = np.array([-1, 0])
    print('8.', cosine_similarity(x, y))


def euclidean_distance(x: np.array, y: np.array):
    return np.sqrt(np.sum((x - y) ** 2))


def hamming_distance(x: np.array, y: np.array, normalize=False):
    factor = 1
    if normalize:
        factor = 1 / len(x)
    distance = factor * np.sum(np.abs(x != y))
    return distance


def cosine_similarity(x: np.array, y: np.array):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


if __name__ == '__main__':
    main()
