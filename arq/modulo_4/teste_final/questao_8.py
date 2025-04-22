import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('fall_detection.csv')
X = df[['TIME','SL','EEG','BP','HR','CIRCLUATION']]
y = df['ACTIVITY']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
print(df.shape)
qtde_test = 0.3 * 16382
print(qtde_test)
