"""
ugly outliers的修复
分类器为linear核的svm分类器
输入数据集为真实数据集
适用于多分类的情况
一、去除训练集中异常值和分类错误的样本后未能提高测试集上分类准确度的可能原因：
1、数据集的复杂性：异常值的比例较小，或者数据本身的分布和特征较为复杂，删除这些样本对整体准确度的提升可能不显著
2、样本的代表性: 如果数据集中的大多数样本是正常的或合理的，即使删除了少量异常样本，模型的学习效果和准确度提升也可能有限
3、模型的表现与数据的关系：如果模型的超参数（如 C 和 gamma）没有经过优化，或数据特征不充分，重新训练的模型可能不会比原始模型有显著的改进
4、异常值的类型: 如果异常值对模型的影响不大，或者异常值在测试集中的表现与训练集中的表现相似，删除这些异常值可能不会对模型的总体性能产生显著影响
5、样本错误: 训练数据中的错误样本可能是由标签错误引起的，而不是模型无法处理的数据特性。如果标签本身不准确，删除这些错误样本可能不会提高模型的性能
6、训练样本的数量: 删除样本可能会导致训练集样本量减少。如果删除的样本比例较高，可能会影响训练集的代表性，进而影响模型的性能。
7、过拟合问题: 如果在处理异常值和错误样本时过度调整训练数据，可能导致模型过拟合，从而在测试集上的性能没有显著提升。
8、评估标准: 使用分类准确度作为性能评估标准可能不总是最合适的，特别是在数据集不平衡的情况下。考虑使用其他评估指标，如F1分数、ROC-AUC等，来更全面地评估模型性能。
二、在类别不均衡的前提下，使用以下指标最合适：
1、F1-Score: 结合了精确率和召回率的调和平均数，适合不均衡数据，因为它对正负样本的比例不敏感。特别适合当你需要平衡精确率和召回率时使用。
2、精确率-召回率曲线（Precision-Recall Curve）: 在类别不均衡时比 ROC 曲线更具信息量，因为它直接关注正类的表现。
3、ROC-AUC: 尽管 ROC-AUC 在类别不均衡时可能会高估模型性能，但它仍然能提供整体性能的良好概览，尤其是当你对假阳性率和真正率的权衡感兴趣时。
4、加权平均指标: 使用加权 F1-score 或加权精确率和召回率，可以对每个类别的表现进行加权，从而适应类别不均衡的情况。
三、对特征进行标准化和编码是有必要的
标准化：确保特征在相似的尺度范围内，减少模型训练的复杂度，提高收敛速度。
分类特征编码：将分类数据转化为数值形式，使模型能够正确地处理这些特征。
训练时间：未经标准化和编码的数据可能会导致 SVM 训练过程变慢，因为模型需要在更复杂的特征空间中进行优化。
"""
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from deepod.models import REPEN, SLAD, ICL, NeuTraL, DeepSAD
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
import numpy as np
from sklearn.metrics import hinge_loss
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier, plot_importance
from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer
import shap
from distfit import distfit
from fitter import Fitter
import scipy.stats as stats
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

def one_hot_encode(y, num_classes):
    y = y.astype(int)
    return np.eye(num_classes)[y]

def calculate_made(data):
    median = np.median(data)  # 计算中位数
    abs_deviation = np.abs(data - median)  # 计算每个数据点与中位数的绝对误差
    mad = np.median(abs_deviation)  # 计算绝对误差均值
    made = 1.843 * mad
    return median, made

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

# section 数据预处理

# subsection 真实数据集且对数据集的特征进行了处理
file_path = "../../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
data = pd.read_excel(file_path)
enc = LabelEncoder()
data['Class'] = enc.fit_transform(data['Class'])
X = data.values[:, :-1]
y = data.values[:, -1]
# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
# 记录原始索引
original_indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.5, random_state=1)
# print("X_train 原始索引:", train_indices)
# print("X_test 原始索引:", test_indices)
class_names = enc.classes_
feature_names = data.columns.values.tolist()
# 将 X 和 y 组合为一个 numpy 数组
combined_array = np.hstack((X, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# 创建新的 DataFrame
data_copy = pd.DataFrame(combined_array, columns=feature_names)
# 对分类特征进行整数编码
categorical_features = [0, 6]
categorical_names = {}
for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data.iloc[:, feature])
    data.iloc[:, feature] = le.transform(data.iloc[:, feature])
    categorical_names[feature] = le.classes_

# subsection 真实数据集且对数据集的特征没有进行标准化和编码处理
# file_path = "../../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
# data = pd.read_excel(file_path)
# enc = LabelEncoder()
# data['Class'] = enc.fit_transform(data['Class'])
# X = data.values[:, :-1]
# y = data.values[:, -1]
# # 记录原始索引
# original_indices = np.arange(len(X))
# X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.5, random_state=1)
# class_names = enc.classes_
# feature_names = data.columns.values.tolist()
# # 创建新的 DataFrame
# data_copy = data.copy()

# subsection 真实数据集且在数据集中人为引入噪声
# file_path = "../../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
# data = pd.read_excel(file_path)
# enc = LabelEncoder()
# data['Class'] = enc.fit_transform(data['Class'])
# X = data.values[:, :-1]
# y = data.values[:, -1]
# # 对不同维度进行标准化
# X = StandardScaler().fit_transform(X)
# # 加入随机噪声
# noise_level = 0.1
# # 计算噪声数量
# n_samples = X.shape[0]
# n_noise = int(noise_level * n_samples)
# # 随机选择要添加噪声的样本
# noise_indices = np.random.choice(n_samples, n_noise, replace=False)
# # 添加高斯噪声到特征
# X[noise_indices] += np.random.normal(0, 1, (n_noise, X.shape[1]))
# # print("Data with noise:\n", X.head())
# # 记录原始索引
# original_indices = np.arange(len(X))
# X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.5, random_state=1)
# feature_names = data.columns.values.tolist()
# combined_array = np.hstack((X, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# # 创建新的 DataFrame
# data_copy = pd.DataFrame(combined_array, columns=feature_names)
# # 训练集中添加了高斯噪声的样本在原始数据集D中的索引
# train_noise = np.intersect1d(train_indices, noise_indices)
# # 测试集中添加了高斯噪声的样本在原始数据集D中的索引
# test_noise = np.intersect1d(test_indices, noise_indices)
# print("训练集中的噪声样本为：", train_noise)
# print("测试集中的噪声样本为：", test_noise)

# SECTION M𝑜 (𝑡, D)
# subsection 针对元组异常的无监督异常检测器GOAD
clf_gold = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf_gold.fit(X_train, y=None)

# SECTION 借助异常检测器，在训练集上进行异常值检测
clf = clf_gold
train_scores = clf.decision_function(X_train)
train_pred_labels, train_confidence = clf.predict(X_train, return_confidence=True)
print("训练集中异常值判定阈值为：", clf.threshold_)
train_outliers_index = []
train_outliers_confidence = []
print("训练集样本数：", len(X_train))
for i in range(len(X_train)):
    if train_pred_labels[i] == 1:
        train_outliers_index.append(i)
        train_outliers_confidence.append(train_confidence[i])
# 训练样本中的异常值索引
print("训练集中异常值索引：", train_outliers_index)
print("训练集中的异常值数量：", len(train_outliers_index))
print("训练集中的异常值的置信度：", train_outliers_confidence)

# SECTION SVM模型的实现
# svm_model = svm.SVC()
svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(X_train, y_train)
train_label_pred = svm_model.predict(X_train)
# 被SVM模型错误分类的样本
# 训练样本中的异常值索引
wrong_classified_indices = np.where(y_train != svm_model.predict(X_train))[0]
# 测试样本中的异常值索引
wrong_test_indices = np.where(y_test != svm_model.predict(X_test))[0]

# SUBSECTION 使用sklearn库中的hinge损失函数
decision_values = svm_model.decision_function(X_train)
predicted_labels = np.argmax(decision_values, axis=1)
# 计算每个样本的hinge损失
num_samples = X_train.shape[0]
num_classes = svm_model.classes_.shape[0]
hinge_losses = np.zeros((num_samples, num_classes))
hinge_loss = np.zeros(num_samples)
for i in range(num_samples):
    correct_class = int(y_train[i])
    for j in range(num_classes):
        if j != correct_class:
            loss_j = max(0, 1 - decision_values[i, correct_class] + decision_values[i, j])
            hinge_losses[i, j] = loss_j
    hinge_loss[i] = np.max(hinge_losses[i])
# 判定异常：设置阈值为 1，超过此值即认为是异常
# 训练样本中的异常值索引
bad_samples = np.where(hinge_loss > 1)[0]
soft_outliers = np.where((hinge_loss > 0) & (hinge_loss <= 1))[0]
correct_samples = np.where(hinge_loss == 0)[0]
# print("损失函数高于损失阈值的样本索引为：", bad_samples)

# subsection 判定训练数据中异常值可能导致分类错误的样本
# 训练数据中的异常值，导致SVM分类错误的样本
inter_outliers = list(set(train_outliers_index) & set(bad_samples))
# 测试数据中的潜在异常值，未导致SVM分类错误，但正确分类的预测值与剩余错误分类的最大预测值相差不足阈值1
inter_soft_outliers = list(set(train_outliers_index) & set(soft_outliers))
# 测试数据中的潜在异常值，未导致SVM分类错误，且正确分类的预测值与剩余错误分类的最大预测值相差超过阈值1
inter_correct_class = list(set(train_outliers_index) & set(correct_samples))
# 训练数据中被SVM误分类的样本与训练数据中的异常值的交集
intersection = np.intersect1d(bad_samples, wrong_classified_indices)
# 训练数据中被SVM误分类的样本中未被判定为异常值的样本
diff_elements = np.setdiff1d(wrong_classified_indices, intersection)
print("分类错误的样本中未被判定为异常值的样本索引：", diff_elements)
print("训练集的异常值中损失函数高于阈值1的样本索引：", inter_outliers)
print("训练集的异常值中损失函数在0和阈值1之间的样本索引：", inter_soft_outliers)
print("训练集的异常值中损失函数为0的样本索引：", inter_correct_class)

# SECTION 原始数据中的svm在各类评价指标下的表现
print("*" * 100)

# subsection 计算SVM的分类准确度
# 准确度是指模型正确分类的样本数占总样本数的比例
print("原始训练集中SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("原始测试集中SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))

# subsection 计算 F1 分数
f1_train = f1_score(y_train, svm_model.predict(X_train), average='weighted')
f1_test = f1_score(y_test, svm_model.predict(X_test), average='weighted')
print("原始训练集SVM分类F1分数：" + str(f1_train))
print("原始测试集SVM分类F1分数：" + str(f1_test))

# SECTION 舍弃掉SVM训练数据中分类错误（hinge损失函数高于1）且被判定为异常值的样本，重新在处理后的训练数据上训练SVM
print("*" * 100)
# subsection 计算 accuracy分数
# 生成布尔索引，为要删除的行创建布尔值数组
mask = np.ones(len(X_train), dtype=bool)
mask[inter_outliers] = False
# 使用布尔索引删除那些既被判定为异常值，又使得hinge损失高于1的样本
X_train_split = X_train[mask]
y_train_split = y_train[mask]
# 重新训练SVM模型
svm_model_split = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model_split.fit(X_train_split, y_train_split)
print("去除异常值中分类错误的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_split, svm_model_split.predict(X_train_split))))
print("去除异常值中分类错误的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_split.predict(X_test))))

print("原SVM模型分错的训练数据中的分类准确度：" +
      str(accuracy_score(y_train[wrong_classified_indices], svm_model.predict(X_train[wrong_classified_indices]))))
print("原SVM模型分错的测试数据中的分类准确度：" +
      str(accuracy_score(y_test[wrong_test_indices], svm_model.predict(X_test[wrong_test_indices]))))
print("去除异常值中分类错误的样本后，重新训练的SVM，在原SVM模型分错的训练数据中的分类准确度：" +
      str(accuracy_score(y_train[wrong_classified_indices], svm_model_split.predict(X_train[wrong_classified_indices]))))
print("去除异常值中分类错误的样本后，重新训练的SVM，在原SVM模型分错的测试数据中的分类准确度：" +
      str(accuracy_score(y_test[wrong_test_indices], svm_model_split.predict(X_test[wrong_test_indices]))))

# subsection 计算 F1 分数
print("*" * 100)
f1_train_split = f1_score(y_train_split, svm_model_split.predict(X_train_split), average='weighted')
f1_test_split = f1_score(y_test, svm_model_split.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类F1分数：" + str(f1_train_split))
print("去除异常值中分类错误的样本后的测试集SVM分类F1分数：" + str(f1_test_split))

print("原SVM模型分错的训练数据中的分类F1分数：" +
      str(f1_score(y_train[wrong_classified_indices], svm_model.predict(X_train[wrong_classified_indices]), average='weighted')))
print("原SVM模型分错的测试数据中的分类F1分数：" +
      str(f1_score(y_test[wrong_test_indices], svm_model.predict(X_test[wrong_test_indices]), average='weighted')))
print("去除异常值中分类错误的样本后，重新训练的SVM，在原SVM模型分错的训练数据中的分类F1分数：" +
      str(f1_score(y_train[wrong_classified_indices], svm_model_split.predict(X_train[wrong_classified_indices]), average='weighted')))
print("去除异常值中分类错误的样本后，重新训练的SVM，在原SVM模型分错的测试数据中的分类F1分数：" +
      str(f1_score(y_test[wrong_test_indices], svm_model_split.predict(X_test[wrong_test_indices]), average='weighted')))

# SECTION 舍弃掉SVM训练数据中hinge损失函数高于1的样本，重新在处理后的训练数据上训练SVM
print("*" * 100)

# subsection 计算 accuracy分数
mask_h = np.ones(len(X_train), dtype=bool)
mask_h[bad_samples] = False
X_train_h = X_train[mask_h]
y_train_h = y_train[mask_h]
svm_model_h = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model_h.fit(X_train_h, y_train_h)
print("去除损失高于阈值的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_h, svm_model_h.predict(X_train_h))))
print("去除损失高于阈值的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_h.predict(X_test))))

print("去除损失高于阈值的样本后后，重新训练的SVM，在原SVM模型分错的训练数据中的分类准确度：" +
      str(accuracy_score(y_train[wrong_classified_indices], svm_model_h.predict(X_train[wrong_classified_indices]))))
print("去除损失高于阈值的样本后，重新训练的SVM，在原SVM模型分错的测试数据中的分类准确度：" +
      str(accuracy_score(y_test[wrong_test_indices], svm_model_h.predict(X_test[wrong_test_indices]))))

# subsection 计算 F1 分数
print("*" * 100)
f1_train_h = f1_score(y_train_h, svm_model_h.predict(X_train_h), average='weighted')
f1_test_h = f1_score(y_test, svm_model_h.predict(X_test), average='weighted')
print("去除损失高于阈值的样本后的训练集SVM分类F1分数：" + str(f1_train_h))
print("去除损失高于阈值的样本后后的测试集SVM分类F1分数：" + str(f1_test_h))

print("去除损失高于阈值的样本后，重新训练的SVM，在原SVM模型分错的训练数据中的分类F1分数：" +
      str(f1_score(y_train[wrong_classified_indices], svm_model_h.predict(X_train[wrong_classified_indices]), average='weighted')))
print("去除损失高于阈值的样本后，重新训练的SVM，在原SVM模型分错的测试数据中的分类F1分数：" +
      str(f1_score(y_test[wrong_test_indices], svm_model_h.predict(X_test[wrong_test_indices]), average='weighted')))

# SECTION 舍弃掉SVM训练数据中被判定为异常值的样本，重新在处理后的训练数据上训练SVM
print("*" * 100)

# subsection 计算 accuracy分数
mask_o = np.ones(len(X_train), dtype=bool)
mask_o[train_outliers_index] = False
X_train_o = X_train[mask_o]
y_train_o = y_train[mask_o]
svm_model_o = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model_o.fit(X_train_o, y_train_o)
print("去除判定为异常的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_o, svm_model_o.predict(X_train_o))))
print("去除判定为异常的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_o.predict(X_test))))

print("舍弃掉SVM训练数据中被判定为异常值的样本后，重新训练的SVM，在原SVM模型分错的训练数据中的分类准确度：" +
      str(accuracy_score(y_train[wrong_classified_indices], svm_model_o.predict(X_train[wrong_classified_indices]))))
print("舍弃掉SVM训练数据中被判定为异常值的样本后，重新训练的SVM，在原SVM模型分错的测试数据中的分类准确度：" +
      str(accuracy_score(y_test[wrong_test_indices], svm_model_o.predict(X_test[wrong_test_indices]))))

# subsection 计算 F1 分数
print("*" * 100)
f1_train_o = f1_score(y_train_o, svm_model_o.predict(X_train_o), average='weighted')
f1_test_o = f1_score(y_test, svm_model_o.predict(X_test), average='weighted')
print("舍弃掉SVM训练数据中被判定为异常值的样本后的训练集SVM分类F1分数：" + str(f1_train_o))
print("舍弃掉SVM训练数据中被判定为异常值的样本后的测试集SVM分类F1分数：" + str(f1_test_o))

print("舍弃掉SVM训练数据中被判定为异常值的样本后，重新训练的SVM，在原SVM模型分错的训练数据中的分类F1分数：" +
      str(f1_score(y_train[wrong_classified_indices], svm_model_o.predict(X_train[wrong_classified_indices]), average='weighted')))
print("舍弃掉SVM训练数据中被判定为异常值的样本后，重新训练的SVM，在原SVM模型分错的测试数据中的分类F1分数：" +
      str(f1_score(y_test[wrong_test_indices], svm_model_o.predict(X_test[wrong_test_indices]), average='weighted')))