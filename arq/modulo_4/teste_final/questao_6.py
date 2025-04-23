import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregando o dataset
df = pd.read_csv('fall_detection.csv')

# Configurando o estilo visual
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot para BP
sns.boxplot(ax=axes[0], y=df['BP'], color='skyblue')
axes[0].set_title('Boxplot de BP (Pressão Sanguínea)', fontsize=14)
axes[0].set_ylabel('Valores', fontsize=12)

# Estatísticas descritivas de BP
bp_stats = df['BP'].describe()
axes[0].annotate(f"Mediana: {bp_stats['50%']:.2f}\nMédia: {bp_stats['mean']:.2f}\nMin: {bp_stats['min']:.2f}\nMax: {bp_stats['max']:.2f}",
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Boxplot para HR
sns.boxplot(ax=axes[1], y=df['HR'], color='lightgreen')
axes[1].set_title('Boxplot de HR (Frequência Cardíaca)', fontsize=14)
axes[1].set_ylabel('Valores', fontsize=12)

# Estatísticas descritivas de HR
hr_stats = df['HR'].describe()
axes[1].annotate(f"Mediana: {hr_stats['50%']:.2f}\nMédia: {hr_stats['mean']:.2f}\nMin: {hr_stats['min']:.2f}\nMax: {hr_stats['max']:.2f}",
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Adicionar boxplot por tipo de atividade
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
sns.boxplot(x='ACTIVITY', y='BP', data=df, palette='Set2')
plt.title('Boxplot de BP por Tipo de Atividade', fontsize=14)
plt.xlabel('Tipo de Atividade', fontsize=12)
plt.ylabel('Pressão Sanguínea (BP)', fontsize=12)
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
sns.boxplot(x='ACTIVITY', y='HR', data=df, palette='Set2')
plt.title('Boxplot de HR por Tipo de Atividade', fontsize=14)
plt.xlabel('Tipo de Atividade', fontsize=12)
plt.ylabel('Frequência Cardíaca (HR)', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Apresentar estatísticas descritivas completas
print("Estatísticas descritivas para BP:")
print(bp_stats)
print("\nEstatísticas descritivas para HR:")
print(hr_stats)

# Verificar outliers
def identify_outliers(variable, data):
    Q1 = data[variable].quantile(0.25)
    Q3 = data[variable].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[variable] < lower_bound) | (data[variable] > upper_bound)]
    return outliers, lower_bound, upper_bound, len(outliers)

bp_outliers, bp_lower, bp_upper, bp_count = identify_outliers('BP', df)
hr_outliers, hr_lower, hr_upper, hr_count = identify_outliers('HR', df)

print(f"\nOutliers em BP: {bp_count} valores fora do intervalo [{bp_lower:.2f}, {bp_upper:.2f}]")
print(f"Outliers em HR: {hr_count} valores fora do intervalo [{hr_lower:.2f}, {hr_upper:.2f}]")

# Correlação entre BP e HR
correlation = df['BP'].corr(df['HR'])
print(f"\nCorrelação de Pearson entre BP e HR: {correlation:.4f}")


for col in ['BP', 'HR']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    print(f"Número de outliers em {col}: {outliers.shape[0]}")