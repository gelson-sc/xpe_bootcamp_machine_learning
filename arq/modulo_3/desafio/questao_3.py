import numpy as np

# Definindo o vetor
x = np.array([3, 4])

# Calculando a norma L2 do vetor
norma_l2 = np.linalg.norm(x)

print(f"O vetor x é: {x}")
print(f"A norma L2 do vetor x é: {norma_l2}")

# Verificação manual da norma L2
# A norma L2 é a raiz quadrada da soma dos quadrados dos elementos
norma_l2_manual = np.sqrt(np.sum(x ** 2))
print(f"Verificação manual da norma L2: {norma_l2_manual}")

