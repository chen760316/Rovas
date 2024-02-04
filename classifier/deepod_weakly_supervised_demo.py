# unsupervised methods
from deepod.models.tabular import DevNet
from deepod.utils.data import generate_data
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

"""
DevNet弱监督模型训练和预测
"""

n_train = 500
n_test = 100
n_features = 10
contamination = 0.1
random_state = 42
epochs = 1
hidden_dims = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64

X_train, X_test, y_train, y_test = generate_data(
    n_train=n_train, n_test=n_test, n_features=n_features,
    contamination=contamination, random_state=random_state)
clf = DevNet(epochs=epochs, hidden_dims=hidden_dims, device=device,
                  random_state=random_state)
clf.fit(X_train, y_train)
scores = clf.decision_function(X_test)
pred_labels, confidence = clf.predict(X_test, return_confidence=True)
# train_loader, net, criterion = clf.training_prepare(X_train, y_train)
print("异常值判定阈值为：", clf.threshold_)
print("测试数据分数，预测标签值，预测置信度如下")
for i in range(len(scores)):
    print(scores[i], pred_labels[i], confidence[i])