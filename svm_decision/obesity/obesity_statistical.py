# unsupervised methods
from scipy import stats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
import numpy as np

"""
kaggle datasets
"""
file_path = "../../kaggle_datasets/Obesity_prediction/obesity_data.csv"
obesity = pd.read_csv(file_path)
obesity['ObesityCategory'] = obesity['ObesityCategory'].replace({'Obese': 3, 'Overweight': 2,\
                                                'Normal weight': 1, 'Underweight': 0})
obesity['Gender'] = obesity['Gender'].replace({'Male': 1, 'Female': 0})
obesity_indicates = obesity.drop(["ObesityCategory"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(obesity_indicates, obesity["ObesityCategory"],\
                     test_size=0.5, random_state=1)
X_train = X_train.values
X_test = X_test.values

# 两个简单统计方法中都选择BMI列作为计算源
"""
四分位极差法
"""
q1 = obesity['BMI'].quantile(0.25)
q3 = obesity['BMI'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers_quartile = obesity[(obesity['BMI'] < lower_bound)
                                  | (obesity['BMI'] > upper_bound)]
outliers_index = outliers_quartile.index
"""
Z-分数法
"""
# obesity['BMI_zscore'] = stats.zscore(obesity['BMI'])
# outliers_Z_score = obesity[(obesity['BMI_zscore'] > 3) | (obesity['BMI_zscore'] < -3)]
# print(outliers_Z_score)
# outliers_index = outliers_Z_score.index

print("数据集中异常值索引为：", outliers_index)
print(len(obesity))
print("数据集中异常值所占比例为：", len(outliers_index)/len(obesity))
svm_model = svm.SVC(C=10)
# svm_model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)
print("训练集SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
X_outliers = obesity_indicates.to_numpy()[outliers_index]
y_outliers = obesity['ObesityCategory'].to_numpy()[outliers_index]
print("数据集中outliers的SVM分类准确度：" + str(accuracy_score(y_outliers, svm_model.predict(X_outliers))))