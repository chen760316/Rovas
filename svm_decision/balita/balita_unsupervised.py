# unsupervised methods
from deepod.models import REPEN, SLAD, ICL, NeuTraL
from deepod.models.tabular import GOAD
from deepod.models.tabular import RCA
from deepod.models.tabular import DeepSVDD
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
n_trans = 64
random_state = 42
target_class = 2 
"""
kaggle datasets
"""
file_path = "../../kaggle_datasets/balita/data_balita.csv"
balita = pd.read_csv(file_path)
enc = LabelEncoder()
balita['Nutrition_Status'] = enc.fit_transform(balita['Nutrition_Status'])
balita['Gender'] = enc.fit_transform(balita['Gender'])
# print(balita)
X = balita.values[:,0:3]
y = balita.values[:,3]
# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
"""GOAD"""
# clf = GOAD(epochs=epochs, device=device, n_trans=n_trans)
# clf.fit(X_train, y=None)
"""DeepSVDD"""
# clf = DeepSVDD(epochs=epochs, device=device, random_state=random_state)
# clf.fit(X_train, y=None)
"""RCA"""
# clf = RCA(epochs=epochs, device=device, act='LeakyReLU')
# clf.fit(X_train)
"""RePEN"""
clf = REPEN(epochs=5, device=device)
clf.fit(X_train)
"""SLAD"""
# clf = SLAD(epochs=2, device=device)
# clf.fit(X_train)
"""ICL"""
# clf = ICL(epochs=1, device=device, n_ensemble='auto')
# clf.fit(X_train)
"""NeuTraL"""
# clf = NeuTraL(epochs=1, device=device)
# clf.fit(X_train)

scores = clf.decision_function(X_test)
pred_labels, confidence = clf.predict(X_test, return_confidence=True)
print("训练集中异常值判定阈值为：", clf.threshold_)
# print("测试样本得分，样本标签值，预测置信度如下")
outliers_index = []
test_class_num = 0
for i in range(len(scores)):
    # print(scores[i], pred_labels[i], confidence[i])
    # if pred_labels[i] == 1:
    #     outliers_index.append(i)
    if y_test[i] == target_class:
        test_class_num += 1
        if pred_labels[i] == 1:
            outliers_index.append(i)
# print("测试集中异常值索引为：", outliers_index)
print("测试集中指定类异常点数：",len(outliers_index))
print("测试集中指定类点数：",test_class_num)
print("测试集中指定类异常点所占比例为：", len(outliers_index)/test_class_num)
# print("测试集中异常点所占比例为：", len(outliers_index)/len(scores))
svm_model = svm.SVC(C=10)
# svm_model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)
print("训练集SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
X_test_outliers = X_test[outliers_index]
y_test_outliers_numpy = y_test[outliers_index]
y_test_outliers = pd.Series(y_test_outliers_numpy)
print("测试集中指定类异常点的SVM分类准确度：" + str(accuracy_score(y_test_outliers, svm_model.predict(X_test_outliers))))
