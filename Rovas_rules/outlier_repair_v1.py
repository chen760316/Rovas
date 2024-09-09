"""
ugly outliers的修复
分类器为linear核的svm分类器
一、去除训练集中异常值和分类错误的样本后未能提高测试集上分类准确度的可能原因：
1、数据集的复杂性：异常值的比例较小，或者数据本身的分布和特征较为复杂，删除这些样本对整体准确度的提升可能不显著
2、样本的代表性: 如果数据集中的大多数样本是正常的或合理的，即使删除了少量异常样本，模型的学习效果和准确度提升也可能有限
3、模型的表现与数据的关系：如果模型的超参数（如 C 和 gamma）没有经过优化，或数据特征不充分，重新训练的模型可能不会比原始模型有显著的改进
4、异常值的类型: 如果异常值对模型的影响不大，或者异常值在测试集中的表现与训练集中的表现相似，删除这些异常值可能不会对模型的总体性能产生显著影响
5、样本错误: 训练数据中的错误样本可能是由标签错误引起的，而不是模型无法处理的数据特性。如果标签本身不准确，删除这些错误样本可能不会提高模型的性能
6、训练样本的数量: 删除样本可能会导致训练集样本量减少。如果删除的样本比例较高，可能会影响训练集的代表性，进而影响模型的性能。
7、过拟合问题: 如果在处理异常值和错误样本时过度调整训练数据，可能导致模型过拟合，从而在测试集上的性能没有显著提升。
8、评估标准: 使用分类准确度作为性能评估标准可能不总是最合适的，特别是在数据集不平衡的情况下。考虑使用其他评估指标，如F1分数、ROC-AUC等，来更全面地评估模型性能。
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
file_path = "../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
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
print("训练集样本数：", len(X_train))
for i in range(len(X_train)):
    if train_pred_labels[i] == 1:
        train_outliers_index.append(i)
# 训练样本中的异常值索引
print("训练集中异常值索引：", train_outliers_index)
print("训练集中的异常值数量：", len(train_outliers_index))

# SECTION SVM模型的实现
# svm_model = svm.SVC()
svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(X_train, y_train)
train_label_pred = svm_model.predict(X_train)
# 被SVM模型错误分类的样本
# 训练样本中的异常值索引
wrong_classified_indices = np.where(y_train != svm_model.predict(X_train))[0]

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
print("损失函数高于损失阈值的样本索引为：", bad_samples)

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
# F1分数是精确率和召回率的调和平均值，适用于类别不平衡的情况。
# 精确率（Precision）是正确分类为正类的比例，而召回率（Recall）是所有实际正类中被正确分类的比例。F1分数综合考虑了这两个指标。
# average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
# average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
# average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
# average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。
f1_train = f1_score(y_train, svm_model.predict(X_train), average='weighted')
f1_test = f1_score(y_test, svm_model.predict(X_test), average='weighted')
print("原始训练集SVM分类F1分数：" + str(f1_train))
print("原始测试集SVM分类F1分数：" + str(f1_test))

# subsection 计算ROC-AUC分数
# 对于二分类问题，y_score 是模型对样本为正类的概率估计
# ROC曲线绘制的是不同阈值下的真阳性率（召回率）与假阳性率（1 - 特异性）的关系。
# AUC（Area Under the Curve）则是ROC曲线下的面积，值越接近1表示模型的性能越好
# 确保使用 predict_proba 获取预测概率
y_prob_train = svm_model.predict_proba(X_train)  # 返回每个类别的概率
y_prob_test = svm_model.predict_proba(X_test)
roc_auc_train = roc_auc_score(y_train, y_prob_train, multi_class='ovr')  # 一对多方式
roc_auc_test = roc_auc_score(y_test, y_prob_test, multi_class='ovr')  # 一对多方式
print("原始训练集SVM分类ROC-AUC分数：" + str(roc_auc_train))
print("原始测试集SVM分类ROC-AUC分数：" + str(roc_auc_test))

# subsection 计算精确率
# 精确率是指模型预测为正类的样本中有多少是真正的正类。适用于关注假阳性较少的场景。
# average='micro': 计算全局精确率。对每个样本的预测结果进行汇总，然后计算精确率。
# average='macro': 计算每个类别的精确率，然后取这些精确率的简单平均。适用于不关注类别样本量的情况。
# average='weighted': 计算每个类别的精确率，然后按类别样本量加权平均。适用于类别不平衡的情况。
# average=None: 返回每个类别的精确率，可以帮助你详细了解每个类别的分类性能。
from sklearn.metrics import precision_score
# 计算精确率
precision_train = precision_score(y_train, svm_model.predict(X_train), average='weighted')
precision_test = precision_score(y_test, svm_model.predict(X_test), average='weighted')
print("原始训练集SVM分类精确率：" + str(precision_train))
print("原始测试集SVM分类精确率：" + str(precision_test))

# subsection 召回率 (Recall)
# 召回率是指所有真正的正类中被正确预测为正类的比例。适用于关注假阴性较少的场景。
# average='micro': 计算全局精确率。对每个样本的预测结果进行汇总，然后计算精确率。
# average='macro': 计算每个类别的精确率，然后取这些精确率的简单平均。适用于不关注类别样本量的情况。
# average='weighted': 计算每个类别的精确率，然后按类别样本量加权平均。适用于类别不平衡的情况。
# average=None: 返回每个类别的精确率，可以帮助你详细了解每个类别的分类性能。
from sklearn.metrics import recall_score
# 计算召回率
recall_train = recall_score(y_train, svm_model.predict(X_train), average='weighted')
recall_test = recall_score(y_test, svm_model.predict(X_test), average='weighted')
print("原始训练集SVM分类召回率：" + str(recall_train))
print("原始测试集SVM分类召回率：" + str(recall_test))

# subsection 混淆矩阵 (Confusion Matrix)
# 混淆矩阵提供了真正例、假正例、真负例和假负例的详细计数，帮助理解模型的分类表现。
from sklearn.metrics import confusion_matrix
# 计算混淆矩阵
conf_matrix_train = confusion_matrix(y_train, svm_model.predict(X_train))
conf_matrix_test = confusion_matrix(y_test, svm_model.predict(X_test))
print("原始训练集SVM分类混淆矩阵：")
print(conf_matrix_train)
print("原始测试集SVM分类混淆矩阵：")
print(conf_matrix_test)

# subsection PR曲线 (Precision-Recall Curve)
# PR曲线展示了精确率与召回率之间的权衡，特别适用于处理不平衡数据集。
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
# 计算 PR 曲线
y_pred_train = svm_model.predict(X_train)
y_pred_test = svm_model.predict(X_test)
precision_train, recall_train, _, _ = precision_recall_fscore_support(y_train, y_pred_train, average='macro')
precision_test, recall_test, _, _ = precision_recall_fscore_support(y_test, y_pred_test, average='macro')
# precision_train, recall_train, _ = precision_recall_curve(y_train, svm_model.decision_function(X_train))
# precision_test, recall_test, _ = precision_recall_curve(y_test, svm_model.decision_function(X_test))
# 绘制 PR 曲线
plt.figure()
plt.plot(recall_train, precision_train, color='blue', label='Train PR curve')
plt.plot(recall_test, precision_test, color='red', label='Test PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
# Adding a grid can help in visualization
plt.grid(True)
plt.show()

print("*" * 100)

# SECTION 舍弃掉SVM训练数据中分类错误（hinge损失函数高于1）且被判定为异常值的样本，重新在处理后的训练数据上训练SVM
print("*" * 100)

# 生成布尔索引，为要删除的行创建布尔值数组
mask = np.ones(len(X_train), dtype=bool)
mask[inter_outliers] = False
# 使用布尔索引删除那些既被判定为异常值，又使得hinge损失高于1的样本
X_train_split = X_train[mask]
y_train_split = y_train[mask]
# 重新训练SVM模型
svm_model_split = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model_split.fit(X_train_split, y_train_split)
# 使用训练好的svm模型预测
y_train_pred = svm_model_split.predict(X_train_split)
y_test_pred = svm_model_split.predict(X_test)

# subsection 计算 accuracy分数
print("*" * 100)
print("去除异常值中分类错误的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_split, y_train_pred)))
print("去除异常值中分类错误的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, y_test_pred)))
print("*" * 100)

# subsection 计算 F1 分数
f1_train_split = f1_score(y_train_split, svm_model_split.predict(X_train_split), average='weighted')
f1_test_split = f1_score(y_test, svm_model_split.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类F1分数：" + str(f1_train_split))
print("去除异常值中分类错误的样本后的测试集SVM分类F1分数：" + str(f1_test_split))

# subsection 计算ROC-AUC分数
y_prob_train_split = svm_model_split.predict_proba(X_train_split)  # 返回每个类别的概率
y_prob_test = svm_model_split.predict_proba(X_test)
roc_auc_train = roc_auc_score(y_train_split, y_prob_train_split, multi_class='ovr')  # 一对多方式
roc_auc_test = roc_auc_score(y_test, y_prob_test, multi_class='ovr')  # 一对多方式
print("去除异常值中分类错误的样本后的训练集SVM分类ROC-AUC分数：" + str(roc_auc_train))
print("去除异常值中分类错误的样本后的测试集SVM分类ROC-AUC分数：" + str(roc_auc_test))

# subsection 计算精确率
from sklearn.metrics import precision_score
precision_train = precision_score(y_train_split, svm_model_split.predict(X_train_split), average='weighted')
precision_test = precision_score(y_test, svm_model_split.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类精确率：" + str(precision_train))
print("去除异常值中分类错误的样本后的测试集SVM分类精确率：" + str(precision_test))

# subsection 召回率 (Recall)
from sklearn.metrics import recall_score
recall_train = recall_score(y_train_split, svm_model_split.predict(X_train_split), average='weighted')
recall_test = recall_score(y_test, svm_model_split.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类召回率：" + str(recall_train))
print("去除异常值中分类错误的样本后的测试集SVM分类召回率：" + str(recall_test))

# subsection 混淆矩阵 (Confusion Matrix)
from sklearn.metrics import confusion_matrix
conf_matrix_train = confusion_matrix(y_train_split, svm_model_split.predict(X_train_split))
conf_matrix_test = confusion_matrix(y_test, svm_model_split.predict(X_test))
print("原始训练集SVM分类混淆矩阵：")
print(conf_matrix_train)
print("原始测试集SVM分类混淆矩阵：")
print(conf_matrix_test)

# subsection PR曲线 (Precision-Recall Curve)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
y_pred_train_split = svm_model_split.predict(X_train_split)
y_pred_test = svm_model_split.predict(X_test)
precision_train, recall_train, _, _ = precision_recall_fscore_support(y_train_split, y_pred_train_split, average='macro')
precision_test, recall_test, _, _ = precision_recall_fscore_support(y_test, y_pred_test, average='macro')
plt.figure()
plt.plot(recall_train, precision_train, color='blue', label='Train PR curve')
plt.plot(recall_test, precision_test, color='red', label='Test PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# subsection 计算 Matthews Correlation Coefficient (MCC)分数
from sklearn.metrics import matthews_corrcoef

mcc_train = matthews_corrcoef(y_train_split, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
print(f'训练集的Matthews Correlation Coefficient: {mcc_train}')
print(f'测试集的Matthews Correlation Coefficient: {mcc_test}')

# subsection 计算 Balanced Accuracy分数
from sklearn.metrics import balanced_accuracy_score

balanced_acc_train = balanced_accuracy_score(y_train_split, y_train_pred)
balanced_acc_test = balanced_accuracy_score(y_test, y_test_pred)
print(f'训练集的Balanced Accuracy: {balanced_acc_train}')
print(f'测试集的Balanced Accuracy: {balanced_acc_test}')

# subsection 计算 G-Mean分数
def calculate_gmean(y_true, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 从混淆矩阵中提取 TP、TN、FP、FN
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    # 计算灵敏度和特异性
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    # 计算 G-Mean
    gmean = np.sqrt(sensitivity * specificity)
    return gmean

g_mean_train = calculate_gmean(y_train_split, y_train_pred)
g_mean_test = calculate_gmean(y_test, y_test_pred)
print(f'训练集的g_mean分数: {g_mean_train}')
print(f'测试集的g_mean分数: {g_mean_test}')

# subsection 计算 F-beta Score分数
from sklearn.metrics import fbeta_score

# 设置 beta 参数
beta = 2  # 举例，beta=2 表示召回率的权重是精确度的 2 倍
f_beta_train = fbeta_score(y_train_split, y_train_pred, beta=beta)
f_beta_test = fbeta_score(y_test, y_test_pred, beta=beta)
print(f'训练集的F-beta Score: {f_beta_train}')
print(f'测试集的F-beta Score: {f_beta_test}')

# subsection 计算 Average Precision (AP)分数
from sklearn.metrics import average_precision_score

# 计算 Average Precision
ap_train = average_precision_score(y_train_split, y_train_pred)
ap_test = average_precision_score(y_test, y_test_pred)
print(f'训练集的Average Precision: {ap_train}')
print(f'测试集的Average Precision: {ap_test}')

# subsection 计算 Area Under Precision-Recall Curve (PR AUC)面积
from sklearn.metrics import average_precision_score

pr_auc_train = average_precision_score(y_train_split, y_train_pred)
pr_auc_test = average_precision_score(y_test, y_test_pred)
print(f'训练集的Area Under Precision-Recall Curve (PR AUC): {pr_auc_train}')
print(f'测试集的Area Under Precision-Recall Curve (PR AUC): {pr_auc_test}')

print("*" * 100)

# SECTION 舍弃掉SVM训练数据中分类错误（hinge损失函数高于1）的样本，重新在处理后的训练数据上训练SVM
print("*" * 100)

# subsection 计算 accuracy分数
mask_h = np.ones(len(X_train), dtype=bool)
mask_h[bad_samples] = False
X_train_h = X_train[mask_h]
y_train_h = y_train[mask_h]
svm_model_h = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model_h.fit(X_train_h, y_train_h)
print("*" * 100)
print("去除损失高于阈值的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_h, svm_model_h.predict(X_train_h))))
print("去除损失高于阈值的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_h.predict(X_test))))
print("*" * 100)

# subsection 计算 F1 分数
f1_train_h = f1_score(y_train_h, svm_model_h.predict(X_train_h), average='weighted')
f1_test_h = f1_score(y_test, svm_model_h.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类F1分数：" + str(f1_train_h))
print("去除异常值中分类错误的样本后的测试集SVM分类F1分数：" + str(f1_test_h))

# subsection 计算ROC-AUC分数
y_prob_train_h = svm_model_h.predict_proba(X_train_h)  # 返回每个类别的概率
y_prob_test = svm_model_h.predict_proba(X_test)
roc_auc_train = roc_auc_score(y_train_h, y_prob_train_h, multi_class='ovr')  # 一对多方式
roc_auc_test = roc_auc_score(y_test, y_prob_test, multi_class='ovr')  # 一对多方式
print("去除异常值中分类错误的样本后的训练集SVM分类ROC-AUC分数：" + str(roc_auc_train))
print("去除异常值中分类错误的样本后的测试集SVM分类ROC-AUC分数：" + str(roc_auc_test))

# subsection 计算精确率
from sklearn.metrics import precision_score
precision_train = precision_score(y_train_h, svm_model_h.predict(X_train_h), average='weighted')
precision_test = precision_score(y_test, svm_model_h.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类精确率：" + str(precision_train))
print("去除异常值中分类错误的样本后的测试集SVM分类精确率：" + str(precision_test))

# subsection 召回率 (Recall)
from sklearn.metrics import recall_score
recall_train = recall_score(y_train_h, svm_model_h.predict(X_train_h), average='weighted')
recall_test = recall_score(y_test, svm_model_h.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类召回率：" + str(recall_train))
print("去除异常值中分类错误的样本后的测试集SVM分类召回率：" + str(recall_test))

# subsection 混淆矩阵 (Confusion Matrix)
from sklearn.metrics import confusion_matrix
conf_matrix_train = confusion_matrix(y_train_h, svm_model_h.predict(X_train_h))
conf_matrix_test = confusion_matrix(y_test, svm_model_h.predict(X_test))
print("原始训练集SVM分类混淆矩阵：")
print(conf_matrix_train)
print("原始测试集SVM分类混淆矩阵：")
print(conf_matrix_test)

# subsection PR曲线 (Precision-Recall Curve)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
y_pred_train_h = svm_model_h.predict(X_train_h)
y_pred_test = svm_model_h.predict(X_test)
precision_train, recall_train, _, _ = precision_recall_fscore_support(y_train_h, y_pred_train_h, average='macro')
precision_test, recall_test, _, _ = precision_recall_fscore_support(y_test, y_pred_test, average='macro')
plt.figure()
plt.plot(recall_train, precision_train, color='blue', label='Train PR curve')
plt.plot(recall_test, precision_test, color='red', label='Test PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()

print("*" * 100)

# SECTION 舍弃掉SVM训练数据中被判定为异常值的样本，重新在处理后的训练数据上训练SVM

print("*" * 100)

# subsection 计算 accuracy分数
mask_o = np.ones(len(X_train), dtype=bool)
mask_o[train_outliers_index] = False
X_train_o = X_train[mask_o]
y_train_o = y_train[mask_o]
svm_model_o = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model_o.fit(X_train_o, y_train_o)
print("*" * 100)
print("去除判定为异常的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_o, svm_model_o.predict(X_train_o))))
print("去除判定为异常的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_o.predict(X_test))))
print("*" * 100)

# subsection 计算 F1 分数
f1_train_o = f1_score(y_train_o, svm_model_o.predict(X_train_o), average='weighted')
f1_test_o = f1_score(y_test, svm_model_o.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类F1分数：" + str(f1_train_o))
print("去除异常值中分类错误的样本后的测试集SVM分类F1分数：" + str(f1_test_o))

# subsection 计算ROC-AUC分数
y_prob_train_o = svm_model_o.predict_proba(X_train_o)  # 返回每个类别的概率
y_prob_test = svm_model_o.predict_proba(X_test)
roc_auc_train = roc_auc_score(y_train_o, y_prob_train_o, multi_class='ovr')  # 一对多方式
roc_auc_test = roc_auc_score(y_test, y_prob_test, multi_class='ovr')  # 一对多方式
print("去除异常值中分类错误的样本后的训练集SVM分类ROC-AUC分数：" + str(roc_auc_train))
print("去除异常值中分类错误的样本后的测试集SVM分类ROC-AUC分数：" + str(roc_auc_test))

# subsection 计算精确率
from sklearn.metrics import precision_score
precision_train = precision_score(y_train_o, svm_model_o.predict(X_train_o), average='weighted')
precision_test = precision_score(y_test, svm_model_o.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类精确率：" + str(precision_train))
print("去除异常值中分类错误的样本后的测试集SVM分类精确率：" + str(precision_test))

# subsection 召回率 (Recall)
from sklearn.metrics import recall_score
recall_train = recall_score(y_train_o, svm_model_o.predict(X_train_o), average='weighted')
recall_test = recall_score(y_test, svm_model_o.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类召回率：" + str(recall_train))
print("去除异常值中分类错误的样本后的测试集SVM分类召回率：" + str(recall_test))

# subsection 混淆矩阵 (Confusion Matrix)
from sklearn.metrics import confusion_matrix
conf_matrix_train = confusion_matrix(y_train_o, svm_model_o.predict(X_train_o))
conf_matrix_test = confusion_matrix(y_test, svm_model_o.predict(X_test))
print("原始训练集SVM分类混淆矩阵：")
print(conf_matrix_train)
print("原始测试集SVM分类混淆矩阵：")
print(conf_matrix_test)

# subsection PR曲线 (Precision-Recall Curve)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
y_pred_train_o = svm_model_o.predict(X_train_o)
y_pred_test = svm_model_o.predict(X_test)
precision_train, recall_train, _, _ = precision_recall_fscore_support(y_train_o, y_pred_train_o, average='macro')
precision_test, recall_test, _, _ = precision_recall_fscore_support(y_test, y_pred_test, average='macro')
plt.figure()
plt.plot(recall_train, precision_train, color='blue', label='Train PR curve')
plt.plot(recall_test, precision_test, color='red', label='Test PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()

print("*" * 100)