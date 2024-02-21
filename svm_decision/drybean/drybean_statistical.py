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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

"""
kaggle datasets
"""
file_path = "../../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
drybean = pd.read_excel(file_path)
enc = LabelEncoder()
drybean['Class'] = enc.fit_transform(drybean['Class'])
X = drybean.values[:,0:16]
y = drybean.values[:,16]
X = StandardScaler().fit_transform(X)
indices = np.arange(len(y))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, test_size=0.5, random_state=1)
print(len(idx_test))
target_class = 1

# 两个简单统计方法中都选择ShapeFactor1列作为计算源
"""
四分位极差法
"""
# test_data = drybean.iloc[idx_test]
# q1 = test_data['ShapeFactor1'].quantile(0.25)
# q3 = test_data['ShapeFactor1'].quantile(0.75)
# iqr = q3 - q1

# lower_bound = q1 - 1.5 * iqr
# upper_bound = q3 + 1.5 * iqr

# outliers_quartile = test_data[(test_data['Class'] == target_class) & ((test_data['ShapeFactor1'] < lower_bound) | (test_data['ShapeFactor1'] > upper_bound))]
# print(outliers_quartile)
# outliers_index = outliers_quartile.index
"""
Z-分数法
"""
test_data = drybean.iloc[idx_test]
# test_data = test_data[test_data['Class'] == target_class]
test_data['ShapeFactor1_zscore'] = stats.zscore(test_data['ShapeFactor1'])
outliers_Z_score = test_data[((test_data['ShapeFactor1_zscore'] > 3) | (test_data['ShapeFactor1_zscore'] < -3)) & (test_data['Class'] == target_class)]
# print(outliers_Z_score)
outliers_index = outliers_Z_score.index

print(len(outliers_index))
# print("测试集中指定类异常值索引为：", outliers_index)
test_class_num = len(test_data[test_data['Class'] == target_class])
print("测试集中指定类异常点数：",len(outliers_index))
print("测试集中指定类点数：",test_class_num)
print("测试集中指定类异常值所占比例为：", len(outliers_index)/test_class_num)
svm_model = svm.SVC(C=10)
# svm_model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)
print("训练集SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
X_outliers = X[outliers_index]
y_outliers = y[outliers_index]
print("测试集中outliers的SVM分类准确度：" + str(accuracy_score(y_outliers, svm_model.predict(X_outliers))))