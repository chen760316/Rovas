import nibabel as nib
import torch
import numpy as np
import pandas as pd
from deepod.models.tabular import DeepSAD
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

semi_label_ratio = 0.1

data = pd.read_csv('../datasets/real_outlier/optdigits.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

out_clf = DeepSAD(epochs=epochs, device=device, random_state=random_state)

# 设置弱监督训练样本
# 找到所有标签为 1 的样本索引
positive_indices = np.where(y_train == 1)[0]
# 随机选择 10% 的正样本
n_positive_to_keep = int(len(positive_indices) * semi_label_ratio)
selected_positive_indices = np.random.choice(positive_indices, n_positive_to_keep, replace=False)
# 创建新的标签数组
y_l = np.zeros_like(y_train)  # 默认全为 0
y_l[selected_positive_indices] = 1  # 设置选中的正样本为 1

out_clf.fit(X_train, y=y_l)

test_scores = out_clf.decision_function(X_test)
y_test_pred, test_confidence = out_clf.predict(X_test, return_confidence=True)

"""Accuracy指标"""
print("*" * 100)
print("DeepSAD在测试集中的分类准确度：" + str(accuracy_score(y_test, y_test_pred)))

"""Precision/Recall/F1指标"""
print("*" * 100)

# average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
# average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
# average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
# average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。

print("DeepSAD在测试集中的分类精确度：" + str(precision_score(y_test, y_test_pred, average='weighted')))
print("DeepSAD在测试集中的分类召回率：" + str(recall_score(y_test, y_test_pred, average='weighted')))
print("DeepSAD在测试集中的分类F1分数：" + str(f1_score(y_test, y_test_pred, average='weighted')))

"""ROC-AUC指标"""
print("*" * 100)
roc_auc_test = roc_auc_score(y_test, y_test_pred, multi_class='ovr')  # 一对多方式
print("DeepSAD在测试集中的ROC-AUC分数：" + str(roc_auc_test))

"""PR AUC指标"""
print("*" * 100)
# 计算预测概率
y_scores = 1 / (1 + np.exp(-test_scores))
# 计算 Precision 和 Recall
precision, recall, _ = precision_recall_curve(y_test, y_scores)
# 计算 PR AUC
pr_auc = auc(recall, precision)
print("PR AUC 分数:", pr_auc)

"""AP指标"""
print("*" * 100)
# 计算预测概率
y_scores = 1 / (1 + np.exp(-test_scores))
# 计算 Average Precision
ap_score = average_precision_score(y_test, y_scores)
print("AP分数:", ap_score)