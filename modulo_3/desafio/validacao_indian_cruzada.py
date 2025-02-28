import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
df = pd.read_csv('processed_indian_liver_patient.csv')
print(df.head(10))
# print(df.describe())
# print(df.shape)
# print(df.isnull().sum())
#{'Female': 0, 'Male': 1}
class_counts = df['Class'].value_counts()
print("Contagem de valores na coluna 'Class' # 1 -> Vivos 2 -> Mortos:")
print(class_counts)
# # Passo 2: Verificar dados categóricos e valores faltantes
# print("\nVerificação de dados categóricos e valores faltantes:")
# print(df.info())
# print("\nValores ausentes por coluna:")
# print(df.isna().sum())
X = df.drop('Class', axis=1)
y = df['Class']
print(X.head(3))
print(y.head(3))
print('Amostras e Features:', df.shape)

#print(df.columns)

name_to_class = {'vivo': 1,'morto': 2}
classifier_cv = RandomForestClassifier(n_estimators= 10, random_state=42)
scores_cv = cross_val_score(classifier_cv, X, y, cv=5)
print(scores_cv)
print("Acurácia: %0.2f (+/- %0.2f)" % (scores_cv.mean(), scores_cv.std() * 2))

scores_cv_precision = cross_val_score(classifier_cv, X, y, cv=5, scoring='precision')
scores_cv_recall = cross_val_score(classifier_cv, X, y, cv=5, scoring='recall')
scores_cv_f1 = cross_val_score(classifier_cv, X, y, cv=5, scoring='f1')

print("Precision: %0.2f (+/- %0.2f)" % (scores_cv_precision.mean(), scores_cv_precision.std() * 2))
print("Recall: %0.2f (+/- %0.2f)" % (scores_cv_recall.mean(), scores_cv_recall.std() * 2))
print("F1: %0.2f (+/- %0.2f)" % (scores_cv_f1.mean(), scores_cv_f1.std() * 2))

# validar
classifier_cv = RandomForestClassifier(n_estimators= 10, random_state=42)

scores_cv = cross_val_score(classifier_cv, X, y, cv=10)
scores_cv_precision = cross_val_score(classifier_cv, X, y, cv=10, scoring='precision')
scores_cv_recall = cross_val_score(classifier_cv, X, y, cv=10, scoring='recall')
scores_cv_f1 = cross_val_score(classifier_cv, X, y, cv=10, scoring='f1')

print("Acurácia: %0.2f (+/- %0.2f)" % (scores_cv.mean(), scores_cv.std() * 2))
print("Precision: %0.2f (+/- %0.2f)" % (scores_cv_precision.mean(), scores_cv_precision.std() * 2))
print("Recall: %0.2f (+/- %0.2f)" % (scores_cv_recall.mean(), scores_cv_recall.std() * 2))
print("F1: %0.2f (+/- %0.2f)" % (scores_cv_f1.mean(), scores_cv_f1.std() * 2))