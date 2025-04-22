import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('fall_detection.csv')
X = df.drop(columns=['ACTIVITY'])
y = df['ACTIVITY']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Questão 9
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"9. Logistic Regression Accuracy: {accuracy_lr}")

# Questão 10:
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train_scaled, y_train)
y_pred_dtc = dtc.predict(X_test_scaled)
accuracy_dtc = accuracy_score(y_test, y_pred_dtc)
print(f"10. Decision Tree Accuracy: {accuracy_dtc}")

# Questão 11
rfc = RandomForestClassifier(n_estimators=50, random_state=42)
rfc.fit(X_train_scaled, y_train)
y_pred_rfc = rfc.predict(X_test_scaled)
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
print(f"11. Random Forest Accuracy: {accuracy_rfc}")

# Questão 12
gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, max_features=6, max_depth=5, random_state=42)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"12. Gradient Boosting Accuracy: {accuracy_gb}")

# Questão 13: Confusion Matrix
# KNN
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
confusion_knn = confusion_matrix(y_test, y_pred_knn)

# SVM
clf_svm = SVC(gamma='auto', kernel='rbf', random_state=42)
clf_svm.fit(X_train_scaled, y_train)
y_pred_svm = clf_svm.predict(X_test_scaled)
confusion_svm = confusion_matrix(y_test, y_pred_svm)

# Logistic Regression
confusion_lr = confusion_matrix(y_test, y_pred_lr)

# Decision Tree
confusion_dtc = confusion_matrix(y_test, y_pred_dtc)

# Analizar a matriz de confusão ACTIVITY=3
fall_activity_index = 3 - 1  # Adjusting for zero-based indexing
print(f"KNN Confusion Matrix:\n{confusion_knn}")
print(f"SVM Confusion Matrix:\n{confusion_svm}")
print(f"Logistic Regression Confusion Matrix:\n{confusion_lr}")
print(f"Decision Tree Confusion Matrix:\n{confusion_dtc}")

# Extract percentage of correct predictions for ACTIVITY=3
knn_correct = confusion_knn[fall_activity_index, fall_activity_index] / sum(confusion_knn[fall_activity_index]) * 100
svm_correct = confusion_svm[fall_activity_index, fall_activity_index] / sum(confusion_svm[fall_activity_index]) * 100
lr_correct = confusion_lr[fall_activity_index, fall_activity_index] / sum(confusion_lr[fall_activity_index]) * 100
dtc_correct = confusion_dtc[fall_activity_index, fall_activity_index] / sum(confusion_dtc[fall_activity_index]) * 100

print(f"Percentage of correct predictions for ACTIVITY=3:")
print(f"KNN: {knn_correct}%")
print(f"SVM: {svm_correct}%")
print(f"Logistic Regression: {lr_correct}%")
print(f"Decision Tree: {dtc_correct}%")
