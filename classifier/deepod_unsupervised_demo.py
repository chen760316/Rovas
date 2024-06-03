# unsupervised methods
from deepod.models import RCA
from deepod.models.tabular import GOAD
from deepod.utils.data import generate_data
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import sys

sys.path.append('E:/xuhongzuo/Rovas/')
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
# print("测试数据分数：", scores)
# print("*" * 50)
# print("预测标签：", pred_labels)
# print("预测置信度：", confidence)

"""
kaggle datasets
"""
file_path = "../kaggle_datasets/Apple_Quality/apple_quality.csv"
apple_quality = pd.read_csv(file_path)
apple_quality['Quality'] = apple_quality['Quality'].replace({'good': 1, 'bad': 0})
apple_indicates = apple_quality.drop(["Quality", "A_id"], axis=1)
X_train_Kaggle, X_test_Kaggle, y_train_Kaggle, y_test_Kaggle = \
    train_test_split(apple_indicates, apple_quality["Quality"],
                     test_size=0.2, random_state=1)
svm_model = svm.SVC(C=10)
svm_model.fit(X_train_Kaggle, y_train_Kaggle)
print("SVM训练模型评分：" + str(accuracy_score(y_train_Kaggle, svm_model.predict(X_train_Kaggle))))
print("SVM测试模型评分：" + str(accuracy_score(y_test_Kaggle, svm_model.predict(X_test_Kaggle))))

