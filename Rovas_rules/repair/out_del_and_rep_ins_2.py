import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from deepod.models.tabular import GOAD
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 在特征中注入高斯随机噪声
file_path = "../../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
data = pd.read_excel(file_path)
enc = LabelEncoder()
# 原始数据集D对应的Dataframe
data['Class'] = enc.fit_transform(data['Class'])
X = data.values[:, :-1]
y = data.values[:, -1]
# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
# 记录原始索引
original_indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.2, random_state=1)
# 加入随机噪声
noise_level = 0.2
# 计算噪声数量
n_samples = X.shape[0]
n_noise = int(noise_level * n_samples)
# 随机选择要添加噪声的样本
noise_indices = np.random.choice(n_samples, n_noise, replace=False)
# 添加高斯噪声到特征
X_copy = np.copy(X)
X_copy[noise_indices] += np.random.normal(0, 1, (n_noise, X.shape[1]))
# 从含噪数据中生成训练数据和测试数据
X_train_copy = X_copy[train_indices]
X_test_copy = X_copy[test_indices]
feature_names = data.columns.values.tolist()
combined_array = np.hstack((X_copy, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# 添加噪声后的数据集D'对应的Dataframe
data_copy = pd.DataFrame(combined_array, columns=feature_names)
# 训练集中添加了高斯噪声的样本在原始数据集D中的索引
train_noise = np.intersect1d(train_indices, noise_indices)
# 测试集中添加了高斯噪声的样本在原始数据集D中的索引
test_noise = np.intersect1d(test_indices, noise_indices)
# print("训练集中的噪声样本为：", train_noise)
# print("测试集中的噪声样本为：", test_noise)

# SECTION M𝑜 (𝑡, D) 针对元组异常的无监督异常检测器GOAD
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42
clf_gold = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf_gold.fit(X_train, y=None)

# SECTION 借助异常检测器，在训练集上进行异常值检测。
#  经过检验，加入高斯噪声会影响异常值判别

# subsection 从原始数据集的训练集和测试集中检测出的异常值

out_clf = clf_gold

# 从原始训练集中检测出异常值索引

train_scores = out_clf.decision_function(X_train)
train_pred_labels, train_confidence = out_clf.predict(X_train, return_confidence=True)
print("训练集中异常值判定阈值为：", out_clf.threshold_)
train_outliers_index = []
print("训练集样本数：", len(X_train))
for i in range(len(X_train)):
    if train_pred_labels[i] == 1:
        train_outliers_index.append(i)

# 从原始测试集中检测出异常值索引

test_scores = out_clf.decision_function(X_test)
test_pred_labels, test_confidence = out_clf.predict(X_test, return_confidence=True)
print("测试集中异常值判定阈值为：", out_clf.threshold_)
test_outliers_index = []
print("测试集样本数：", len(X_test))
for i in range(len(X_test)):
    if test_pred_labels[i] == 1:
        test_outliers_index.append(i)

# subsection 从加噪数据集的训练集和测试集中检测出的异常值

clf_gold_noise = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf_gold_noise.fit(X_train_copy, y=None)
out_clf_noise = clf_gold_noise

# 从加噪训练集中检测出异常值索引

train_scores_noise = out_clf_noise.decision_function(X_train_copy)
train_pred_labels_noise, train_confidence_noise = out_clf_noise.predict(X_train_copy, return_confidence=True)
print("加噪训练集中异常值判定阈值为：", out_clf_noise.threshold_)
train_outliers_index_noise = []
print("加噪训练集样本数：", len(X_train_copy))
for i in range(len(X_train_copy)):
    if train_pred_labels_noise[i] == 1:
        train_outliers_index_noise.append(i)

# 从加噪测试集中检测出异常值索引

test_scores_noise = out_clf_noise.decision_function(X_test_copy)
test_pred_labels_noise, test_confidence_noise = out_clf_noise.predict(X_test_copy, return_confidence=True)
print("加噪测试集中异常值判定阈值为：", out_clf_noise.threshold_)
test_outliers_index_noise = []
print("加噪测试集样本数：", len(X_test_copy))
for i in range(len(X_test_copy)):
    if test_pred_labels_noise[i] == 1:
        test_outliers_index_noise.append(i)

# SECTION 借助异常检测器，确定加噪之后的样本是否更容易被检测为异常值
#  经过检验，噪声数据的异常值比例远高于正常训练数据中的异常值比例

# 在原始数据集D中对应的索引
set_train_noise = set(train_noise)
set_outliers_index_noise = set(train_indices[train_outliers_index_noise])
noise_intersection = set_train_noise & set_outliers_index_noise
print("加噪训练集中的异常值比例：", len(train_outliers_index_noise)/len(X_train_copy))
print("加噪训练集中，噪声数据的异常值比例：", len(noise_intersection)/len(set_train_noise))

# SECTION SVM模型的实现

# subsection 原始数据集上训练的SVM模型在训练集和测试集中分错的样本比例

print("*" * 100)
svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(X_train, y_train)
train_label_pred = svm_model.predict(X_train)

# 训练样本中被SVM模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != svm_model.predict(X_train))[0]
print("训练样本中被SVM模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被SVM模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != svm_model.predict(X_test))[0]
print("测试样本中被SVM模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被SVM模型错误分类的样本
print("完整数据集D中被SVM模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

# subsection 加噪数据集上训练的SVM模型在训练集和测试集中分错的样本比例

print("*" * 100)
svm_model_noise = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model_noise.fit(X_train_copy, y_train)
train_label_pred_noise = svm_model_noise.predict(X_train_copy)

# 加噪训练样本中被SVM模型错误分类的样本
wrong_classified_train_indices_noise = np.where(y_train != svm_model_noise.predict(X_train_copy))[0]
print("加噪训练样本中被SVM模型错误分类的样本占总加噪训练样本的比例：", len(wrong_classified_train_indices_noise)/len(y_train))

# 加噪测试样本中被SVM模型错误分类的样本
wrong_classified_test_indices_noise = np.where(y_test != svm_model_noise.predict(X_test_copy))[0]
print("加噪测试样本中被SVM模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices_noise)/len(y_test))

# 整体加噪数据集D中被SVM模型错误分类的样本
print("完整数据集D中被SVM模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise))/(len(y_train) + len(y_test)))

# section 识别X_copy中需要修复的元组

# 异常检测器检测出的训练集和测试集中的异常值在原含噪数据D'中的索引
train_outliers_noise = train_indices[train_outliers_index_noise]
test_outliers_noise = test_indices[test_outliers_index_noise]
outliers_noise = np.union1d(train_outliers_noise, test_outliers_noise)

# 在加噪数据集D'上训练的SVM模型，其分类错误的样本在原含噪数据D'中的索引
train_wrong_clf_noise = train_indices[wrong_classified_train_indices_noise]
test_wrong_clf_noise = test_indices[wrong_classified_test_indices_noise]
wrong_clf_noise = np.union1d(train_wrong_clf_noise, test_wrong_clf_noise)

# outliers和分错样本的并集
train_union = np.union1d(train_outliers_noise, train_wrong_clf_noise)
test_union = np.union1d(test_outliers_noise, test_wrong_clf_noise)

# 加噪数据集D'上需要修复的值
# 需要修复的特征和标签值
X_copy_repair_indices = np.union1d(outliers_noise, wrong_clf_noise)
X_copy_repair = X_copy[X_copy_repair_indices]
y_repair = y[X_copy_repair_indices]

# 生成保留的行索引
rows_to_keep = np.setdiff1d(np.arange(X_copy.shape[0]), X_copy_repair_indices)

# 使用保留的行索引选择D'中的正常数据
# 无需修复的特征和标签值
X_copy_inners = X_copy[rows_to_keep]
y_inners = y[rows_to_keep]

# section 对X_copy中需要修复的元组进行特征修复

# subsection 确定有影响力的特征

# choice 使用系数方法
# # 找到训练数据中有影响力的特征（最佳特证数是4，7，8，10）
# top_k_svm = 4
#
# # 提取系数
# feature_importances_coef = np.abs(svm_model_noise.coef_[0])
#
# # 对系数进行排序
# top_k_indices = np.argpartition(-feature_importances_coef, top_k_svm)[:top_k_svm]
# print("SVM模型选择的特征索引是：", top_k_indices)

# choice 使用SelectFromModel方法(效果好)
# # 使用SelectFromModel来选择重要特征
# sfm = SelectFromModel(svm_model_noise, threshold='mean', prefit=True)
# X_selected = sfm.transform(X_copy)
# # X_selected = sfm.transform(X_train_copy)
#
# # 获取选择的特征索引
# top_k_indices = sfm.get_support(indices=True)
#
# # 打印选择的特征索引
# print("SelectFromModel选择器选择的特征索引:", top_k_indices)

# choice 借助方差判别有影响力的特征(效果不好)
# top_k_var = 8
# variances = np.var(X_copy, axis=0)
# top_k_indices = np.argsort(variances)[-top_k_var:][::-1]
# print("方差最大的前{}个特征的索引：{}".format(top_k_var, top_k_indices))

# choice sklearn库的SelectKBest选择器，借助Fisher检验筛选最有影响力的k个特征(效果不好)
# top_k_fisher = 4
# selector = SelectKBest(score_func=f_classif, k=top_k_fisher)
# y_trans_fisher = y.reshape(-1)
# X_new = selector.fit_transform(X_copy, y_trans_fisher)
# # 获取被选中的特征的索引
# top_k_indices = selector.get_support(indices=True)
# print("SelectKBest选择器借助Fisher检验的最有影响力的{}个特征的索引：{}".format(top_k_fisher, top_k_indices))

# choice 无模型(非参数)方法中的Permutation Feature Importance-slearn (效果好)(速度慢)
# top_k_svm = 8
# result = permutation_importance(svm_model_noise, X_train_copy, y_train, n_repeats=10,random_state=42)
# permutation_importance = result.importances_mean
# top_k_indices = np.argpartition(-permutation_importance, top_k_svm)[:top_k_svm]
# print("Permutation Feature Importance-slearn检验的最有影响力的的前{}个属性的索引：{}".format(top_k_svm, top_k_indices))

# choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)（速度快）
# 特征数取4或6
i = 16
top_k_svm = 5
np.random.seed(1)
categorical_features = [0, 6]
categorical_names = {}
for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data.iloc[:, feature])
    data.iloc[:, feature] = le.transform(data.iloc[:, feature])
    categorical_names[feature] = le.classes_
explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=feature_names,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3)
# predict_proba 方法用于分类任务，predict 方法用于回归任务
predict_fn = lambda x: svm_model.predict_proba(x)
exp = explainer.explain_instance(X_train[i], predict_fn, num_features=6)
# 获取最具影响力的特征及其权重
top_features = exp.as_list()
important_features = []
for feature_set in top_features:
    feature_long = feature_set[0]
    for feature in feature_names:
        if set(feature).issubset(set(feature_long)):
            important_features.append(feature)
            break
top_k_indices = [feature_names.index(feature_name) for feature_name in important_features]
print("LIME检验的最有影响力的的前{}个属性的索引：{}".format(top_k_svm, top_k_indices))

# subsection 确定异常值，使用2MADe方法确定有影响力的列中的异常值，将异常值的部分设置为np.nan
for i in range(X_copy.shape[1]):
    if i in top_k_indices:
        column_data = X_copy[:, i]
        # 计算中位数
        median = np.median(column_data)
        mean = np.mean(column_data)
        # 计算每个数据点的绝对偏差
        mad = np.median(np.abs(column_data - median))
        # 2MADe 判定阈值
        threshold = 2 * mad
        # 标记异常值
        outliers = np.abs(column_data - median) > threshold
        # 获取 outliers 中元素为 True 的下标索引
        outlier_indices = np.where(outliers)[0]

        # 计算交集
        # intersection = np.intersect1d(outlier_indices, X_copy_repair_indices)
        # X_copy[intersection, i] = np.nan

        # 将所有需要修复的行对应的列位置的元素替换为均值
        intersection = X_copy_repair_indices
        X_copy[intersection, i] = mean


# subsection 使用knn修复所有被标记为nan的异常特征

# 创建 KNN Imputer 对象
knn_imputer = KNNImputer(n_neighbors=3)

# 使用 KNN 算法填补异常特征
X_copy = knn_imputer.fit_transform(X_copy)
X_train_copy = X_copy[train_indices]
X_test_copy = X_copy[test_indices]

# subsection 重新在修复后的数据上训练SVM模型

svm_repair = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_repair.fit(X_train_copy, y_train)
y_train_pred = svm_repair.predict(X_train_copy)
y_test_pred = svm_repair.predict(X_test_copy)

print("*" * 100)
# 训练样本中被SVM模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
print("加噪标签修复后，训练样本中被SVM模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被SVM模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
print("加噪标签修复后，测试样本中被SVM模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被SVM模型错误分类的样本
print("加噪标签修复后，完整数据集D中被SVM模型错误分类的样本占总完整数据的比例：",
      (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))