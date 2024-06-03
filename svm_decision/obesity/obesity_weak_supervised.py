# weaksupervised methods
from deepod.models import REPEN, SLAD, ICL, RDP, NeuTraL, DeepSAD, FeaWAD, RoSAS, PReNet
from deepod.models.tabular import DevNet
from deepod.models.tabular import GOAD
from deepod.models.tabular import RCA
from deepod.models.tabular import DeepSVDD
from deepod.utils.data import generate_data
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import LabelEncoder

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42
hidden_dims=20
np.random.seed(42)
"""
kaggle datasets
"""
file_path = "../../kaggle_datasets/Obesity_prediction/obesity_data.csv"
obesity = pd.read_csv(file_path)
enc = LabelEncoder()
obesity['ObesityCategory'] = enc.fit_transform(obesity['ObesityCategory'])
obesity['Gender'] = obesity['Gender'].replace({'Male': 1, 'Female': 0})
# obesity_indicates = obesity.drop(["ObesityCategory"], axis=1)
X = obesity.values[:,0:6]
y = obesity.values[:,6]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

"""DevNet"""
clf = DevNet(epochs=1, hidden_dims=20, device=device,
                          random_state=42)
anom_id = np.where(y_train == 1)[0]
known_anom_id = np.random.choice(anom_id, 10, replace=False)
y_semi = np.zeros_like(y_train, dtype=int)
y_semi[known_anom_id] = 1
clf.fit(X_train, y_semi)
"""DeepSAD"""
# clf = DeepSAD(epochs=1, hidden_dims=20,
#                    device=device,
#                    random_state=42)
# anom_id = np.where(y_train == 1)[0]
# known_anom_id = np.random.choice(anom_id, 10, replace=False)
# y_semi = np.zeros_like(y_train, dtype=int)
# y_semi[known_anom_id] = 1
# clf.fit(X_train, y_semi)
"""RoSAS"""
# clf = RoSAS(epochs=1, hidden_dims=20, device=device, random_state=42)
# anom_id = np.where(y_train == 1)[0]
# known_anom_id = np.random.choice(anom_id, 10, replace=False)
# y_semi = np.zeros_like(y_train, dtype=int)
# y_semi[known_anom_id] = 1
# clf.fit(X_train, y_semi)
"""PReNet"""
# clf = PReNet(epochs=1,
#                   epoch_steps=20,
#                   device=device,
#                   batch_size=256,
#                   lr=1e-5)
# anom_id = np.where(y_train == 1)[0]
# known_anom_id = np.random.choice(anom_id, 10, replace=False)
# y_semi = np.zeros_like(y_train, dtype=int)
# y_semi[known_anom_id] = 1
# clf.fit(X_train, y_semi)

scores = clf.decision_function(X_test)
pred_labels, confidence = clf.predict(X_test, return_confidence=True)
print("训练集中异常值判定阈值为：", clf.threshold_)
# print("测试样本得分，样本标签值，预测置信度如下")
# outliers_index = []
# for i in range(len(scores)):
#     print(scores[i], pred_labels[i], confidence[i])
#     if pred_labels[i] == 1:
#         outliers_index.append(i)
outliers_index = np.where(np.array(pred_labels) == 1)[0]
print("测试集中异常值索引为：", outliers_index)
preds=clf.predict(X_test)
print(preds[outliers_index])
print("测试集中异常值所占比例为：", len(outliers_index)/len(scores))
svm_model = svm.SVC(C=10)
# svm_model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)
print("训练集SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
X_test_outliers = X_test[outliers_index]
# y_test_outliers_numpy = y_test.to_numpy()[outliers_index]
# y_test_outliers = pd.Series(y_test_outliers_numpy)
print("测试集中outliers的SVM分类准确度：" + str(accuracy_score(y_test[outliers_index], svm_model.predict(X_test_outliers))))
