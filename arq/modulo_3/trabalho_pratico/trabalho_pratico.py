import numpy as np


def main():
    pass


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
