from deepod.models.tabular import RCA
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from deepod.models.tabular import DevNet

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

# SECTION 数据预处理
# file_path = "../kaggle_datasets/balita/data_balita.csv"
# data = pd.read_csv(file_path)
# # 从 data 中随机抽取 10% 的样本
# data = data.sample(frac=0.1, random_state=42)
# enc = LabelEncoder()
# # 原始数据集D对应的Dataframe
# data['Nutrition_Status'] = enc.fit_transform(data['Nutrition_Status'])
# data['Gender'] = enc.fit_transform(data['Gender'])
# categorical_features = [0, 1]
# X = data.values[:, :-1]
# y = data.values[:, -1]

# file_path = "../kaggle_datasets/Apple_Quality/apple_quality.csv"
# data = pd.read_csv(file_path)
# # 删除id列
# data = data.drop(data.columns[0], axis=1)
# enc = LabelEncoder()
# # 原始数据集D对应的Dataframe
# data['Quality'] = enc.fit_transform(data['Quality'])
# categorical_features = []
# X = data.values[:, :-1]
# y = data.values[:, -1]

# choice drybean数据集(效果好)
# file_path = "../normal_experiment/datasets/multi_class/drybean.xlsx"
# data = pd.read_excel(file_path)

# choice obesity数据集(效果好)
# file_path = "../normal_experiment/datasets/multi_class/obesity.csv"
# data = pd.read_csv(file_path)

# choice balita数据集(SVM拟合效果差，但修复后效果提升显著)
# file_path = "../normal_experiment/datasets/multi_class/balita.csv"
# data = pd.read_csv(file_path)

# choice apple数据集(效果提升小)
file_path = "../normal_experiment/datasets/multi_class/apple.csv"
data = pd.read_csv(file_path)

enc = LabelEncoder()
label_name = data.columns[-1]

# 原始数据集D对应的Dataframe
data[label_name] = enc.fit_transform(data[label_name])

# 检测非数值列
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

# 为每个非数值列创建一个 LabelEncoder 实例
encoders = {}
for column in non_numeric_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    encoders[column] = encoder  # 保存每个列的编码器，以便将来可能需要解码

X = data.values[:, :-1]
y = data.values[:, -1]

X = StandardScaler().fit_transform(X)
# 记录原始索引
original_indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.3, random_state=1)

# SECTION 训练异常检测器
# choice RCA异常检测器
clf = RCA(epochs=epochs, device=device, act='LeakyReLU')
clf.fit(X_train)

# # choice DevNet异常检测器
# epochs = 1
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# n_trans = 64
# random_state = 42
# hidden_dims = 20
# epoch_steps = 20
# batch_size = 256
# lr = 1e-5
#
# # 统计不同值及其数量
# unique_values, counts = np.unique(y, return_counts=True)
#
# # 找到最小标签的数量
# min_count = counts.min()
# total_count = counts.sum()
#
# # 计算比例
# proportion = min_count / total_count
# min_count_index = np.argmin(counts)  # 找到最小数量的索引
# min_label = unique_values[min_count_index]  # 对应的标签值
# # 设置弱监督训练样本
# # 找到所有标签为 1 的样本索引
# semi_label_ratio = 0.1  # 设置已知的异常标签比例
# positive_indices = np.where(y_train == min_label)[0]
# # 随机选择 10% 的正样本
# n_positive_to_keep = int(len(positive_indices) * semi_label_ratio)
# selected_positive_indices = np.random.choice(positive_indices, n_positive_to_keep, replace=False)
# # 创建用于异常检测器的训练标签
# y_semi = np.zeros_like(y_train)  # 默认全为 0
# y_semi[selected_positive_indices] = 1  # 设置选中的正样本为 1
# # 创建用于异常检测器的测试标签
# y_semi_test = np.zeros_like(y_test)
# test_positive_indices = np.where(y_test == min_label)[0]
# y_semi_test[test_positive_indices] = 1
# clf = DevNet(epochs=epochs, hidden_dims=hidden_dims, device=device,
#                           random_state=random_state)
# clf.fit(X_train, y_semi)

# SECTION 检测测试集中的异常值
test_scores = clf.decision_function(X_test)
test_pred_labels, test_confidence = clf.predict(X_test, return_confidence=True)
print("测试集中异常值判定阈值为：", clf.threshold_)
test_outliers_index = []
print("测试集样本数：", len(X_test))
for i in range(len(X_test)):
    if test_pred_labels[i] == 1:
        test_outliers_index.append(i)
# 训练样本中的异常值索引
print("测试集中异常值索引：", test_outliers_index)
print("测试集中的异常值数量：", len(test_outliers_index))

# SECTION 训练SVM分类器
# svm_model = svm.SVC()
svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(X_train, y_train)
test_label_pred = svm_model.predict(X_test)
wrong_classified_indices = np.where(y_test != svm_model.predict(X_test))[0]
print("测试集中分类错误的样本数量：", len(wrong_classified_indices))

# SECTION 使用sklearn库中的hinge损失函数

# choice 自定义计算hinge损失
# decision_values = svm_model.decision_function(X_test)
# y_pred = np.argmax(decision_values, axis=1)
# # 计算每个样本的hinge损失
# num_samples = X_test.shape[0]
# num_classes = svm_model.classes_.shape[0]
# hinge_losses = np.zeros((num_samples, num_classes))
# hinge_loss = np.zeros(num_samples)
# for i in range(num_samples):
#     correct_class = int(y_test[i])
#     for j in range(num_classes):
#         if j != correct_class:
#             loss_j = max(0, 1 - decision_values[i, correct_class] + decision_values[i, j])
#             hinge_losses[i, j] = loss_j
#     hinge_loss[i] = np.max(hinge_losses[i])
# bad_samples = np.where(hinge_loss > 1)[0]
# # 测试样本中的bad outliers索引
# bad_outliers_index = np.intersect1d(test_outliers_index, bad_samples)
# print("检测出的outliers中bad outliers的数量：", len(bad_outliers_index))

# choice 计算交叉熵损失
# decision_values = svm_model.decision_function(X_test)
# # 应用 Softmax 函数
# y_pred = softmax(decision_values, axis=1)
# # 创建 OneHotEncoder 实例
# encoder = OneHotEncoder(sparse=False)
# # 拟合并转换 y_test
# y_true = encoder.fit_transform(y_test.reshape(-1, 1))
# # 计算每个样本的损失
# loss_per_sample = -np.sum(y_true * np.log(y_pred + 1e-12), axis=1)
# # 计算测试集平均多分类交叉熵损失
# average_loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
# bad_samples = np.where(loss_per_sample > average_loss)[0]
# good_samples = np.where(loss_per_sample <= average_loss)[0]
# # 测试样本中的bad outliers索引
# bad_outliers_index = np.intersect1d(test_outliers_index, bad_samples)
# print("检测出的outliers中bad outliers的数量：", len(bad_outliers_index))

# choice 使用二元交叉熵损失函数
def binary_cross_entropy(y_true, y_pred):
    # 防止对数计算中的零
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    losses = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    loss = np.mean(losses)
    return losses, loss

def sigmoid(decision_values):
    return 1 / (1 + np.exp(-decision_values))

decision_values = svm_model.decision_function(X_test)
y_prediction = svm_model.predict(X_test)
# 计算概率
probabilities = sigmoid(decision_values)
print("属于类 1 的概率:", probabilities)
print("属于类 0 的概率:", 1 - probabilities)  # 属于类 0 的概率

loss_per_sample, average_loss = binary_cross_entropy(y_test, probabilities)
good_samples = np.where(loss_per_sample > average_loss)[0]
bad_samples = np.where(loss_per_sample <= average_loss)[0]
# 测试样本中的bad outliers索引
bad_outliers_index = np.intersect1d(test_outliers_index, bad_samples)
good_outliers_index = np.intersect1d(test_outliers_index, good_samples)
print("检测出的outliers中bad outliers的数量：", len(bad_outliers_index))
print("测试集中好的样本数量：", len(good_samples))
print("测试集中坏的样本数量：", len(bad_samples))
print("坏的outliers占总的坏的样本的比例为：", len(bad_outliers_index)/len(bad_samples))
bad_intersection = np.intersect1d(wrong_classified_indices, bad_outliers_index)
print("误分类样本中同时出现在bad outliers中的样本，被所有误分类样本包含的比例：", len(bad_intersection)/len(wrong_classified_indices))
good_intersection = np.intersect1d(wrong_classified_indices, good_outliers_index)
print("误分类样本中同时出现在good outliers中的样本，被所有误分类样本包含的比例：", len(good_intersection)/len(wrong_classified_indices))
print("*"*100)

# section 统计好的和坏的离群值中的统计指标

# 测试样本中的good outliers索引
good_outliers_index = np.intersect1d(test_outliers_index, good_samples)
print("检测出的outliers中good outliers的数量：", len(good_outliers_index))
# good outliers中分错的比例
good_wrong_indies = []
for i in good_outliers_index:
    true_label = y_test[i]
    if true_label != test_label_pred[i]:
        good_wrong_indies.append(i)
print("good outliers中样本分错的比例：", len(good_wrong_indies)/len(good_outliers_index))
# bad outliers中分错的比例
bad_wrong_indies = []
for i in bad_outliers_index:
    true_label = y_test[i]
    if true_label != test_label_pred[i]:
        bad_wrong_indies.append(i)
print("bad outliers中样本分错的比例：", len(bad_wrong_indies)/len(bad_outliers_index))

# section 测试集中离群值中被SVM分类器误分类的数量
wrong_classify_indices = []
for i in test_outliers_index:
    true_label = y_test[i]
    if true_label != test_label_pred[i]:
        wrong_classify_indices.append(i)
print("检测出的outliers中被SVM分类器误分类的数量(ugly outliers的数量)：", len(wrong_classify_indices))