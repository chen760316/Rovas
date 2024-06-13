import pandas as pd
import numpy as np
import winreg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
###################
import sys

sys.path.append('E:/xuhongzuo/Rovas/')

file_path = "../kaggle_datasets/Apple_Quality/apple_quality.csv"
apple_quality = pd.read_csv(file_path)
apple_quality['Quality'] = apple_quality['Quality'].replace({'good': 1, 'bad': 0})
missing_values = apple_quality.isnull()
# print(missing_values)
print(apple_quality.info())
print(apple_quality.head())
apple_indicates = apple_quality.drop(["Quality", "A_id"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(apple_indicates, apple_quality["Quality"], test_size=0.2, random_state=1)
# svm_linear = svm.LinearSVC()
svm_linear = svm.SVC(C=10)
# svm_linear = svm.SVC(C=1, kernel="linear", decision_function_shape="ovr")
# svm_rbf = svm.SVC(C=1, kernel="rbf", decision_function_shape="ovr")
# svm_poly = svm.SVC(C=1, kernel="poly", decision_function_shape="ovr")
# svm_sigmoid = svm.SVC(C=1, kernel="sigmoid", decision_function_shape="ovr")
svm_model = svm_linear
svm_model.fit(X_train, y_train)
print("SVM训练模型评分：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("SVM测试模型评分：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
