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
    cosine_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return cosine_sim


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
    centered_cosine_sim = cosine_similarity(x, y)
    return centered_cosine_sim


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

# c. Similaridade de cossenos entre U1 e U4
questao_c = cosine_similarity(U1, U4)
print(f'C:{questao_c:.2f}')

# d. Similaridade de cossenos entre U1 e U5
questao_d = cosine_similarity(U1, U5)
print(f'D:{questao_d:.2f}')

# e. Similaridade de cossenos centralizada entre U1 e U2
questao_e = centered_cosine_similarity(U1, U2)
print(f'E:{questao_e:.2f}')

# f. Similaridade de cossenos centralizada entre U1 e U3
questao_f = centered_cosine_similarity(U1, U3)
print(f'F:{questao_f:.2f}')

# g. Similaridade de cossenos centralizada entre U1 e U4
questao_g = centered_cosine_similarity(U1, U4)
print(f'G:{questao_g:.2f}')

# h. Similaridade de cossenos centralizada entre U1 e U5
questao_h = centered_cosine_similarity(U1, U5)
print(f'H:{questao_h:.2f}')

# i. e j. Encontrando usuários mais similares a U1 usando similaridade centralizada
similaridades = {
    'U2': questao_e,
    'U3': questao_f,
    'U4': questao_g,
    'U5': questao_h
}

sorted_similaridades = sorted(similaridades.items(), key=lambda x: x[1], reverse=True)
print(f"Usuários em ordem de similaridade com U1: {sorted_similaridades}")

mais_similar = sorted_similaridades[0][0]
segundo_mais_similar = sorted_similaridades[1][0]

print(f"I: O usuário mais similar a U1 é {mais_similar}")
print(f"J: O segundo usuário mais similar a U1 é {segundo_mais_similar}")

# k. Predição User-User para (U1, I2)
# Pegando os 2 usuários mais similares
vizinhos_k2 = sorted_similaridades[:2]
vizinhos_ids = [vizinho[0] for vizinho in vizinhos_k2]

# Calculando as avaliações do item I2 para os vizinhos
avaliacoes_i2 = [ratings.loc[vizinho, 'I2'] for vizinho in vizinhos_ids]
# Filtrando apenas avaliações não nulas
avaliacoes_i2_nonzero = [av for av in avaliacoes_i2 if av > 0]
# Calculando a média simples
predicao_u1_i2 = np.mean(avaliacoes_i2_nonzero) if len(avaliacoes_i2_nonzero) > 0 else 0
print(f"K: A predição para (U1, I2) é {predicao_u1_i2:.2f}")

# l. Predição User-User para (U1, I1) - considerando que avaliação é desconhecida
# Para simular que U1,I1 é desconhecida, vamos calcular de novo as similaridades sem esse valor
ratings_temp = ratings.copy()
# Salvando o valor original para posterior verificação
valor_original_u1_i1 = ratings_temp.loc['U1', 'I1']
# Definindo como 0 (desconhecido)
ratings_temp.loc['U1', 'I1'] = 0


# Precisamos recalcular as similaridades centralizadas
def calc_similaridades_sem_u1i1():
    U1_temp = ratings_temp.loc['U1'].values

    # Calculando similaridades centralizadas
    sim_u1_u2 = centered_cosine_similarity(U1_temp, U2)
    sim_u1_u3 = centered_cosine_similarity(U1_temp, U3)
    sim_u1_u4 = centered_cosine_similarity(U1_temp, U4)
    sim_u1_u5 = centered_cosine_similarity(U1_temp, U5)

    similaridades_temp = {
        'U2': sim_u1_u2,
        'U3': sim_u1_u3,
        'U4': sim_u1_u4,
        'U5': sim_u1_u5
    }

    return sorted(similaridades_temp.items(), key=lambda x: x[1], reverse=True)


sorted_similaridades_l = calc_similaridades_sem_u1i1()
vizinhos_l = sorted_similaridades_l[:2]
vizinhos_ids_l = [vizinho[0] for vizinho in vizinhos_l]

# Calculando as avaliações do item I1 para os vizinhos
avaliacoes_i1 = [ratings.loc[vizinho, 'I1'] for vizinho in vizinhos_ids_l]
# Filtrando apenas avaliações não nulas
avaliacoes_i1_nonzero = [av for av in avaliacoes_i1 if av > 0]
# Calculando a média simples
predicao_u1_i1 = np.mean(avaliacoes_i1_nonzero) if len(avaliacoes_i1_nonzero) > 0 else 0
print(f"L: A predição para (U1, I1) é {predicao_u1_i1:.2f}")

# m. Erro absoluto da predição (U1, I1)
erro_absoluto = abs(valor_original_u1_i1 - predicao_u1_i1)
print(f"M: O erro absoluto para a predição (U1, I1) é {erro_absoluto:.2f}")

# n. Calculando b_u, b_i e mu para U1 e I2
# Média de todas as avaliações conhecidas (não nulas)
mu = ratings.values[ratings.values > 0].mean()

# Média do usuário U1 (avaliações não nulas)
b_u = ratings.loc['U1'][ratings.loc['U1'] > 0].mean()

# Média do item I2 (avaliações não nulas)
b_i = ratings['I2'][ratings['I2'] > 0].mean()

print(f"N: Para U1 e I2: b_u = {b_u:.2f}, b_i = {b_i:.2f}, mu = {mu:.2f}")

# o. Predição para (U1, I2) usando b_u + b_i - mu
predicao_baseline = b_u + b_i - mu
print(f"O: A predição para (U1, I2) usando o modelo baseline é {predicao_baseline:.2f}")

# Respostas consolidadas
print("\nRespostas Consolidadas:")
print(f"a. Similaridade de cossenos entre U1 e U2: {questao_a:.2f}")
print(f"b. Similaridade de cossenos entre U1 e U3: {questao_b:.2f}")
print(f"c. Similaridade de cossenos entre U1 e U4: {questao_c:.2f}")
print(f"d. Similaridade de cossenos entre U1 e U5: {questao_d:.2f}")
print(f"e. Similaridade de cossenos centralizada entre U1 e U2: {questao_e:.2f}")
print(f"f. Similaridade de cossenos centralizada entre U1 e U3: {questao_f:.2f}")
print(f"g. Similaridade de cossenos centralizada entre U1 e U4: {questao_g:.2f}")
print(f"h. Similaridade de cossenos centralizada entre U1 e U5: {questao_h:.2f}")
print(f"i. Usuário mais similar a U1 (similaridade centralizada): {mais_similar}")
print(f"j. Segundo usuário mais similar a U1 (similaridade centralizada): {segundo_mais_similar}")
print(f"k. Predição para (U1, I2) com User-User e 2 vizinhos: {predicao_u1_i2:.2f}")
print(f"l. Predição para (U1, I1) com User-User e 2 vizinhos (supondo desconhecido): {predicao_u1_i1:.2f}")
print(f"m. Erro absoluto para a predição (U1, I1): {erro_absoluto:.2f}")
print(f"n. Para U1 e I2: b_u = {b_u:.2f}, b_i = {b_i:.2f}, mu = {mu:.2f}")
print(f"o. Predição para (U1, I2) usando o modelo baseline: {predicao_baseline:.2f}")