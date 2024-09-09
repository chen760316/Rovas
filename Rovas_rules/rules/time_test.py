from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import time

# 示例数据
data = pd.DataFrame({
    'feature1': np.random.rand(1000),
    'feature2': np.random.choice([10, 100, 200], 1000),
    'feature3': np.random.rand(1000),
    'target': np.random.choice([0, 1], 1000)
})

X = data.drop('target', axis=1)
y = data['target']

# 未进行预处理的数据
start_time = time.time()
model = SVC()
model.fit(X, y)
print("未进行标准化和编码的训练时间:", time.time() - start_time)

# 进行标准化和编码的数据
X_standardized = StandardScaler().fit_transform(X[['feature1', 'feature3']])
categorical_features = ['feature2']
for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

X_encoded_standardized = pd.concat([pd.DataFrame(X_standardized, columns=['feature1', 'feature3']), X[categorical_features]], axis=1)

start_time = time.time()
model.fit(X_encoded_standardized, y)
print("进行标准化和编码的训练时间:", time.time() - start_time)
