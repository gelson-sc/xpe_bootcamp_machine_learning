import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

bloodtransf_df = pd.read_csv('bloodtransf.csv')

print(bloodtransf_df.head())
print("\nBlood Transfusion Dataset:")
print(bloodtransf_df.shape)
print("\nDados faltantes no Blood Transfusion Dataset:")
print(bloodtransf_df.isnull().sum())

# Blood Transfusion Dataset
X_blood = bloodtransf_df.drop('Class', axis=1)
y_blood = bloodtransf_df['Class']
X_train_blood, X_test_blood, y_train_blood, y_test_blood = train_test_split(X_blood, y_blood, test_size=0.37, random_state=5762)

model_blood = SVC(kernel='rbf', probability=True)
model_blood.fit(X_train_blood, y_train_blood)

y_pred_blood = model_blood.predict(X_test_blood)
y_pred_proba_blood = model_blood.predict_proba(X_test_blood)[:, 1]

# Avaliar as métricas
accuracy = accuracy_score(y_test_blood, y_pred_blood)
precision = precision_score(y_test_blood, y_pred_blood)
recall = recall_score(y_test_blood, y_pred_blood)
f1 = f1_score(y_test_blood, y_pred_blood)
roc_auc = roc_auc_score(y_test_blood, y_pred_proba_blood)

print(f"Acurácia: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUROC: {roc_auc}")
