"""
xuhongzuo库中的无监督异常检测模型使用
未区分训练集和测试集
"""
from deepod.models import REPEN, SLAD, ICL, NeuTraL
from deepod.models.tabular import GOAD
from deepod.models.tabular import RCA
from deepod.models.tabular import DeepSVDD
from deepod.models import DeepSAD, RoSAS, PReNet
from deepod.models.tabular import DevNet
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.datasets import make_blobs

"""
outliers_1检测出的异常值数量: 131
outliers_2检测出的异常值数量: 105
outliers_3检测出的异常值数量: 112
outliers_4检测出的异常值数量: 98
outliers_5检测出的异常值数量: 105
outliers_6检测出的异常值数量: 105
outliers_7检测出的异常值数量: 105
"""

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

# 借助make_blobs库合成数据
feature_number = 5
X, _ = make_blobs(n_samples=1000, centers=3, n_features=feature_number, random_state=None)

# 添加噪声
file_path = "../kaggle_datasets/Apple_Quality/apple_quality.csv"
apple_quality = pd.read_csv(file_path)
new_rows = apple_quality.iloc[:50, 1:6].values

# X_change = np.vstack((X, new_rows))[:, 0]
# X_change = X_change.reshape(-1, 1)
X_change = np.vstack((X, new_rows))

"""GOAD（检测出49个）"""
clf_1 = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf_1.fit(X_change, y=None)
preds_1 = clf_1.predict(X_change)

"""DeepSVDD（全检测出）"""
clf_2 = DeepSVDD(epochs=epochs, device=device, random_state=random_state)
clf_2.fit(X_change, y=None)
preds_2 = clf_2.predict(X_change)

"""RCA(检测精度低，没检出来)"""
clf_3 = RCA(epochs=epochs, device=device, act='LeakyReLU')
clf_3.fit(X_change)
preds_3 = clf_3.predict(X_change)

"""RePEN（全检测出）"""
clf_4 = REPEN(epochs=5, device=device)
clf_4.fit(X_change)
preds_4 = clf_4.predict(X_change)

"""SLAD（检出来49个）"""
clf_5 = SLAD(epochs=2, device=device)
clf_5.fit(X_change)
preds_5 = clf_5.predict(X_change)

"""ICL（检出来37个）"""
clf_6 = ICL(epochs=1, device=device, n_ensemble='auto')
clf_6.fit(X_change)
preds_6 = clf_6.predict(X_change)

"""NeuTraL(全检出来)"""
clf_7 = NeuTraL(epochs=1, device=device)
clf_7.fit(X_change)
preds_7 = clf_7.predict(X_change)

outliers_1 = np.where(preds_1 == 1)[0]
outliers_2 = np.where(preds_2 == 1)[0]
outliers_3 = np.where(preds_3 == 1)[0]
outliers_4 = np.where(preds_4 == 1)[0]
outliers_5 = np.where(preds_5 == 1)[0]
outliers_6 = np.where(preds_6 == 1)[0]
outliers_7 = np.where(preds_7 == 1)[0]

outliers_list = [outliers_1, outliers_2, outliers_3, outliers_4, outliers_5, outliers_6, outliers_7]
# outliers_list = [outliers_1, outliers_2, outliers_3, outliers_4, outliers_7]

for i, outliers in enumerate(outliers_list, start=1):
    print(f"outliers_{i}: {outliers}")

for i, outliers in enumerate(outliers_list, start=1):
    print(f"outliers_{i}检测出的异常值数量: {len(outliers)}")