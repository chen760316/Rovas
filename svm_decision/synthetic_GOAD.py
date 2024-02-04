# unsupervised methods
from deepod.models.tabular import GOAD
from deepod.utils.data import generate_data
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""
GOAD无监督模型训练和预测
"""

n_train = 500
n_test = 100
n_features = 10
contamination = 0.1
random_state = 42
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64

X_train, X_test, y_train, y_test = generate_data(
    n_train=n_train, n_test=n_test, n_features=n_features,
    contamination=contamination, random_state=random_state)
clf = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf.fit(X_train, y=None)
scores = clf.decision_function(X_test)
pred_labels, confidence = clf.predict(X_test, return_confidence=True)
# train_loader, net, criterion = clf.training_prepare(X_train, y_train)
print("异常值判定阈值为：", clf.threshold_)
print("测试数据分数，预测标签值，预测置信度如下")
for i in range(len(scores)):
    print(scores[i], pred_labels[i], confidence[i])
outliers_index = []
for i in range(len(scores)):
    print(scores[i], pred_labels[i], confidence[i])
    if pred_labels[i] == 1:
        outliers_index.append(i)
print("测试集中异常值索引为：", outliers_index)
print("测试集中异常值所占比例为：", len(outliers_index)/len(scores))
svm_model = svm.SVC(C=10)
svm_model.fit(X_train, y_train)
print("训练集SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
X_test_outliers = X_test[outliers_index]
y_test_outliers = y_test[outliers_index]
print("测试集中outliers的SVM分类准确度：" + str(accuracy_score(y_test_outliers, svm_model.predict(X_test_outliers))))