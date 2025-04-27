# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', 1000)
# Definindo configurações para melhor visualização
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set(style='whitegrid')

# Carregando o dataset
print("Carregando o dataset...")
df_cartoes = pd.read_csv('/home/gelson/datasets/creditcard.csv')

# ============== ANÁLISE EXPLORATÓRIA DOS DADOS (EDA) ===============

print("\n==== Análise Exploratória de Dados ====")

# Visão geral do dataset
print("\nInformações do dataset:")
print(df_cartoes.info())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df_cartoes.describe())

# Verificando valores ausentes
print("\nValores ausentes por coluna:")
print(df_cartoes.isnull().sum())

# Análise de distribuição da variável de classe
print("\nDistribuição das classes:")
print(df_cartoes['Class'].value_counts())
print(
    f"Porcentagem de fraudes: {df_cartoes['Class'].value_counts()[1] / df_cartoes['Class'].value_counts().sum() * 100:.4f}%")

# Visualizando a distribuição das classes
# plt.figure(figsize=(10, 6))
# sns.countplot(x='Class', data=df_cartoes)
# plt.title('Distribuição de Classes (Fraude vs Não Fraude)')
# plt.ylabel('Contagem')
# plt.show()

# Análise da distribuição do valor das transações
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# sns.histplot(df_cartoes['Amount'], kde=True)
# plt.title('Distribuição do Valor das Transações')
# plt.xlabel('Valor')
# plt.ylabel('Frequência')
#
# plt.subplot(1, 2, 2)
# sns.boxplot(x='Class', y='Amount', data=df_cartoes)
# plt.title('Valor das Transações por Classe')
# plt.ylabel('Valor')
# plt.xlabel('Classe (0: Normal, 1: Fraude)')
# plt.tight_layout()
# plt.show()

# Análise temporal das transações
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# sns.histplot(df_cartoes['Time'], kde=True)
# plt.title('Distribuição do Tempo das Transações')
# plt.xlabel('Tempo (segundos)')
# plt.ylabel('Frequência')
#
# plt.subplot(1, 2, 2)
# plt.scatter(df_cartoes['Time'], df_cartoes['Amount'], c=df_cartoes['Class'], cmap='coolwarm', alpha=0.5)
# plt.title('Relação entre Tempo e Valor das Transações')
# plt.xlabel('Tempo (segundos)')
# plt.ylabel('Valor')
# plt.colorbar(label='Classe')
# plt.tight_layout()
# plt.show()

# Análise de correlação entre variáveis
# plt.figure(figsize=(18, 12))
# correlation_matrix = df_cartoes.corr()
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
# plt.title('Matriz de Correlação')
# plt.tight_layout()
# plt.show()

# Análise das variáveis V1 a V28 em relação à classe
# plt.figure(figsize=(20, 40))
# for i in range(1, 29):
#     plt.subplot(7, 4, i)
#     sns.boxplot(x='Class', y=f'V{i}', data=df_cartoes)
#     plt.title(f'V{i} por Classe')
# plt.tight_layout()
# plt.show()

# ============== PREPARAÇÃO DOS DADOS ===============

print("\n==== Preparação dos Dados ====")

# Normalização dos dados (Amount e Time)
scaler = StandardScaler()
df_cartoes_ajustado = df_cartoes.copy()
df_cartoes_ajustado['scaled_amount'] = scaler.fit_transform(df_cartoes_ajustado[['Amount']])
df_cartoes_ajustado['scaled_time'] = scaler.fit_transform(df_cartoes_ajustado[['Time']])

# Removendo as colunas originais Time e Amount
df_cartoes_ajustado = df_cartoes_ajustado.drop(['Time', 'Amount'], axis=1)

# Separando as variáveis de entrada e saída
entrada = df_cartoes_ajustado.drop('Class', axis=1)
saida = df_cartoes_ajustado['Class']

# Normalizando todas as variáveis de entrada
scaler_features = StandardScaler()
entrada_normalizada = scaler_features.fit_transform(entrada)

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(entrada_normalizada, saida,
                                                    test_size=0.3, random_state=42)

print(f"Formato do conjunto de treinamento: {X_train.shape}")
print(f"Formato do conjunto de teste: {X_test.shape}")

# ============== COMPARAÇÃO E AJUSTE DE MODELOS DE CLASSIFICAÇÃO ===============

print("\n==== Modelos de Classificação com Dados Desbalanceados ====")


# Função para avaliar o desempenho do modelo
def avaliar_modelo(modelo, X_train, X_test, y_train, y_test, nome_modelo):
    # Treinamento do modelo
    modelo.fit(X_train, y_train)

    # Predições
    y_pred = modelo.predict(X_test)

    # Métricas de avaliação
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred)
    revocacao = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Relatório de classificação
    report = classification_report(y_test, y_pred)

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    # Curva ROC
    if hasattr(modelo, "predict_proba"):
        y_prob = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = None, None, None

    # Visualizações
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.xlabel('Predito')
    plt.ylabel('Real')

    if fpr is not None and tpr is not None:
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title(f'Curva ROC - {nome_modelo}')
        plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    print(f"\nResultados para {nome_modelo}:")
    print(f"Acurácia: {acuracia:.4f}")
    print(f"Precisão: {precisao:.4f}")
    print(f"Revocação: {revocacao:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nRelatório de Classificação:")
    print(report)

    return acuracia, precisao, revocacao, f1


# 1. K-means
print("\n=== Modelo K-means ===")
k_means_pca = KMeans(n_clusters=4, random_state=42)
k_means_pca.fit(X_train)

# Como K-means é um algoritmo não supervisionado, não usamos a função de avaliação padrão
# Vamos avaliar o agrupamento
labels = k_means_pca.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.scatter(k_means_pca.cluster_centers_[:, 0], k_means_pca.cluster_centers_[:, 1], c='red', s=200, alpha=0.5)
plt.title('K-means Clustering')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# 2. Regressão Logística
print("\n=== Modelo Regressão Logística ===")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr_acc, lr_prec, lr_rec, lr_f1 = avaliar_modelo(lr, X_train, X_test, y_train, y_test, "Regressão Logística")

# 3. MLP (Multilayer Perceptron)
print("\n=== Modelo MLP (Multilayer Perceptron) ===")
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=(10,), activation='relu', solver='adam', random_state=1)
mlp_acc, mlp_prec, mlp_rec, mlp_f1 = avaliar_modelo(mlp, X_train, X_test, y_train, y_test, "MLP")

# ============== ABORDAGEM COM SUBAMOSTRAGEM (UNDERSAMPLING) ===============

print("\n==== Modelos de Classificação com Subamostragem ====")

# Realizando subamostragem conforme solicitado
n_fraude = len(df_cartoes_ajustado[df_cartoes_ajustado.Class == 1])
indices_fraude = np.array(df_cartoes_ajustado[df_cartoes_ajustado.Class == 1].index)
indices_sem_fraude = np.array(df_cartoes_ajustado[df_cartoes_ajustado.Class == 0].index)

np.random.seed(0)
escolha_sem_fraude = np.random.choice(indices_sem_fraude, n_fraude, replace=False)
indices_subamostragem = np.concatenate([indices_fraude, escolha_sem_fraude], axis=None)
dados_subamostrados = df_cartoes_ajustado.iloc[indices_subamostragem, :]

entradas = dados_subamostrados[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                                'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                                'scaled_amount', 'scaled_time']]
saida = dados_subamostrados['Class']

# Normalizando as entradas
scaler_subamostras = StandardScaler()
entradas_normalizadas = scaler_subamostras.fit_transform(entradas)

# Divisão dos dados balanceados em conjuntos de treinamento e teste
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(entradas_normalizadas, saida,
                                                                    test_size=0.3, random_state=42)

print(f"Formato do conjunto de treinamento balanceado: {X_train_bal.shape}")
print(f"Formato do conjunto de teste balanceado: {X_test_bal.shape}")
print(f"Distribuição das classes no conjunto balanceado: {np.bincount(saida)}")

# 1. Regressão Logística com dados balanceados
print("\n=== Modelo Regressão Logística (Dados Balanceados) ===")
lr_bal = LogisticRegression(max_iter=1000, random_state=42)
lr_bal_acc, lr_bal_prec, lr_bal_rec, lr_bal_f1 = avaliar_modelo(lr_bal, X_train_bal, X_test_bal, y_train_bal,
                                                                y_test_bal, "Regressão Logística (Balanceado)")

# 2. MLP com dados balanceados
print("\n=== Modelo MLP (Dados Balanceados) ===")
mlp_bal = MLPClassifier(alpha=0.001, hidden_layer_sizes=(10,), activation='relu', solver='adam', random_state=1)
mlp_bal_acc, mlp_bal_prec, mlp_bal_rec, mlp_bal_f1 = avaliar_modelo(mlp_bal, X_train_bal, X_test_bal, y_train_bal,
                                                                    y_test_bal, "MLP (Balanceado)")

# ============== COMPARAÇÃO DOS MODELOS ===============

print("\n==== Comparação Final dos Modelos ====")

# Criando um DataFrame para comparar os resultados
modelos = ['Regressão Logística (Desbalanceado)', 'MLP (Desbalanceado)',
           'Regressão Logística (Balanceado)', 'MLP (Balanceado)']
acuracias = [lr_acc, mlp_acc, lr_bal_acc, mlp_bal_acc]
precisoes = [lr_prec, mlp_prec, lr_bal_prec, mlp_bal_prec]
revocacoes = [lr_rec, mlp_rec, lr_bal_rec, mlp_bal_rec]
f1_scores = [lr_f1, mlp_f1, lr_bal_f1, mlp_bal_f1]

comparacao_df = pd.DataFrame({
    'Modelo': modelos,
    'Acurácia': acuracias,
    'Precisão': precisoes,
    'Revocação': revocacoes,
    'F1-Score': f1_scores
})

print("\nComparação dos modelos:")
print(comparacao_df)

# Visualização dos resultados
plt.figure(figsize=(14, 10))

metricas = ['Acurácia', 'Precisão', 'Revocação', 'F1-Score']
cores = ['blue', 'green', 'red', 'purple']

for i, metrica in enumerate(metricas):
    plt.subplot(2, 2, i + 1)
    sns.barplot(x='Modelo', y=metrica, data=comparacao_df, palette='viridis')
    plt.title(f'Comparação de {metrica}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

plt.show()

# Conclusão
print("\n==== Conclusão ====")
print("Análise de desempenho dos modelos:")
melhor_modelo_f1 = comparacao_df.loc[comparacao_df['F1-Score'].idxmax()]
print(
    f"\nO melhor modelo segundo o F1-Score é: {melhor_modelo_f1['Modelo']} com F1-Score de {melhor_modelo_f1['F1-Score']:.4f}")
print(
    f"O melhor modelo para detecção de fraude obteve Precisão de {melhor_modelo_f1['Precisão']:.4f} e Revocação de {melhor_modelo_f1['Revocação']:.4f}")

print("\nObservações sobre os resultados:")
if melhor_modelo_f1['Modelo'].endswith('(Balanceado)'):
    print("- O balanceamento dos dados melhorou significativamente o desempenho do modelo.")
    print(
        "- Para a detecção de fraudes, onde as classes são altamente desbalanceadas, técnicas de balanceamento são essenciais.")
else:
    print("- Mesmo com o desbalanceamento das classes, o modelo conseguiu um bom desempenho.")

print("\nRecomendações:")
print("- Considerar técnicas adicionais como SMOTE ou ajuste de pesos para lidar com o desbalanceamento das classes.")
print("- Explorar técnicas de seleção de características para melhorar o desempenho dos modelos.")
print("- Avaliar o modelo em um contexto de produção, onde o custo de falsos negativos pode ser muito alto.")