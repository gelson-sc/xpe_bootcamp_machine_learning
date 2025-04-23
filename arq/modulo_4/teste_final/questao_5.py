import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv('fall_detection.csv')

# Visualizar as primeiras linhas do dataset para entender sua estrutura
print("Primeiras 5 linhas do dataset:")
print(df.head())

# Verificar informações básicas do dataset
print("\nInformações do dataset:")
print(df.info())

# Calcular a matriz de correlação de Pearson para as variáveis de interesse
variables = ['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']
correlation_matrix = df[variables].corr(method='pearson')

# Mostrar a matriz de correlação
print("\nMatriz de correlação de Pearson:")
print(correlation_matrix)

# Criar uma visualização da matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.4f', linewidths=.5)
plt.title('Matriz de Correlação de Pearson entre Variáveis')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Encontrar as duas variáveis com maior coeficiente de correlação em valor absoluto
# Primeiro transformamos a matriz em um DataFrame com pares de variáveis
corr_pairs = []
for i in range(len(variables)):
    for j in range(i+1, len(variables)):  # evitamos a diagonal e duplicados
        var1 = variables[i]
        var2 = variables[j]
        corr_value = correlation_matrix.iloc[i, j]
        corr_pairs.append((var1, var2, corr_value, abs(corr_value)))

# Converter para DataFrame para facilitar o manuseio
corr_df = pd.DataFrame(corr_pairs, columns=['Variável 1', 'Variável 2', 'Correlação', 'Correlação Absoluta'])

# Ordenar pelo valor absoluto da correlação
corr_df_sorted = corr_df.sort_values('Correlação Absoluta', ascending=False)

# Mostrar os resultados
print("\nPares de variáveis ordenados por valor absoluto da correlação:")
print(corr_df_sorted)

# Identificar o par com maior correlação absoluta
top_pair = corr_df_sorted.iloc[0]
print(f"\nAs duas variáveis com maior coeficiente de correlação em valor absoluto são:")
print(f"{top_pair['Variável 1']} e {top_pair['Variável 2']} com correlação = {top_pair['Correlação']:.4f} (valor absoluto = {top_pair['Correlação Absoluta']:.4f})")

# Visualizar a relação entre as duas variáveis mais correlacionadas
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x=top_pair['Variável 1'], y=top_pair['Variável 2'], hue='ACTIVITY')
plt.title(f'Relação entre {top_pair["Variável 1"]} e {top_pair["Variável 2"]}')
plt.tight_layout()
plt.savefig('top_correlation_scatter.png')
plt.close()