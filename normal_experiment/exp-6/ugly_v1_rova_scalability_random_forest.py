"""
𝑅(𝑡) ∧ outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃) ∧ loss(M, D, 𝑡) > 𝜆 ∧ M𝑐 (𝑅, 𝐴,M) → ugly(𝑡)
测试ugly v1 outliers检测规则的可扩展性（重点关注谓词的可扩展性）
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from deepod.models.tabular import GOAD

from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
import warnings
warnings.filterwarnings("ignore")
import time
from memory_profiler import memory_usage
import tracemalloc
from contextlib import contextmanager

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

@contextmanager
def memory_timer():
    tracemalloc.start()  # 开始跟踪内存
    yield  # 允许执行代码块
    current, peak = tracemalloc.get_traced_memory()  # 获取当前和峰值内存
    tracemalloc.stop()  # 停止跟踪内存
    print(f"Current Memory Usage: {current / 10**6:.2f} MiB")
    print(f"Peak Memory Usage: {peak / 10**6:.2f} MiB")

with memory_timer():
    # section 标准数据集处理

    start_time = time.time()  # 开始时间

    # subsection 含有不同异常比例的真实数据集

    # choice Annthyroid数据集(效果稳定)
    # file_path = "../datasets/real_outlier_varying_ratios/Annthyroid/Annthyroid_02_v01.csv"
    # file_path = "../datasets/real_outlier_varying_ratios/Annthyroid/Annthyroid_05_v01.csv"
    # file_path = "../datasets/real_outlier_varying_ratios/Annthyroid/Annthyroid_07.csv"

    # choice Cardiotocography数据集(效果稳定)
    # file_path = "../datasets/real_outlier_varying_ratios/Cardiotocography/Cardiotocography_02_v01.csv"
    # file_path = "../datasets/real_outlier_varying_ratios/Cardiotocography/Cardiotocography_05_v01.csv"
    # file_path = "../datasets/real_outlier_varying_ratios/Cardiotocography/Cardiotocography_10_v01.csv"
    # file_path = "../datasets/real_outlier_varying_ratios/Cardiotocography/Cardiotocography_20_v01.csv"
    # file_path = "../datasets/real_outlier_varying_ratios/Cardiotocography/Cardiotocography_22.csv"

    # choice PageBlocks数据集(效果稳定)
    # file_path = "../datasets/real_outlier_varying_ratios/PageBlocks/PageBlocks_02_v01.csv"
    # file_path = "../datasets/real_outlier_varying_ratios/PageBlocks/PageBlocks_05_v01.csv"

    # choice Wilt数据集(效果稳定)
    # file_path = "../datasets/real_outlier_varying_ratios/Wilt/Wilt_02_v01.csv"
    # file_path = "../datasets/real_outlier_varying_ratios/Wilt/Wilt_05.csv"

    # subsection 含有不同异常类型和异常比例的合成数据集（从真实数据中加入噪声合成）

    # choice Annthyroid数据集+cluster噪声+不同噪声比例(效果稳定)
    # file_path = "../datasets/synthetic_outlier/annthyroid_cluster_0.1.csv"
    # file_path = "../datasets/synthetic_outlier/annthyroid_cluster_0.2.csv"
    # file_path = "../datasets/synthetic_outlier/annthyroid_cluster_0.3.csv"

    # choice Cardiotocography数据集+local噪声+不同噪声比例(好用)
    # file_path = "../datasets/synthetic_outlier/Cardiotocography_local_0.1.csv"
    # file_path = "../datasets/synthetic_outlier/Cardiotocography_local_0.2.csv"
    # file_path = "../datasets/synthetic_outlier/Cardiotocography_local_0.3.csv"

    # choice PageBlocks数据集+global噪声+不同噪声比例(效果稳定)
    # file_path = "../datasets/synthetic_outlier/PageBlocks_global_0.1.csv"
    # file_path = "../datasets/synthetic_outlier/PageBlocks_global_0.2.csv"
    # file_path = "../datasets/synthetic_outlier/PageBlocks_global_0.3.csv"

    # choice satellite数据集+local噪声+不同噪声比例(好用)
    # file_path = "../datasets/synthetic_outlier/satellite_0.1.csv"
    # file_path = "../datasets/synthetic_outlier/satellite_0.2.csv"
    # file_path = "../datasets/synthetic_outlier/satellite_0.3.csv"

    # choice annthyroid数据集+local噪声+不同噪声比例(好用)
    # file_path = "../datasets/synthetic_outlier/annthyroid_0.1.csv"
    # file_path = "../datasets/synthetic_outlier/annthyroid_0.2.csv"
    file_path = "../datasets/synthetic_outlier/annthyroid_0.3.csv"

    data = pd.read_csv(file_path)
    # 如果数据量超过20000行，就随机采样到20000行
    if len(data) > 20000:
        data = data.sample(n=20000, random_state=42)

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

    # 找到分类特征的列名
    categorical_columns = data.select_dtypes(exclude=['float']).columns[:-1]
    # 获取分类特征对应的索引
    categorical_features = [data.columns.get_loc(col) for col in categorical_columns]

    all_columns = data.columns.values.tolist()
    feature_names = all_columns[:-1]
    class_name = all_columns[-1]

    # 统计不同值及其数量
    unique_values, counts = np.unique(y, return_counts=True)

    # 输出结果
    for value, count in zip(unique_values, counts):
        print(f"标签: {value}, 数量: {count}")

    # 找到最小标签的数量
    min_count = counts.min()
    total_count = counts.sum()

    # 计算比例
    proportion = min_count / total_count
    print(f"较少标签占据的比例: {proportion:.4f}")
    min_count_index = np.argmin(counts)  # 找到最小数量的索引
    min_label = unique_values[min_count_index]  # 对应的标签值

    end_time = time.time()  # 结束时间
    print("数据集标准处理耗时：", end_time - start_time)

    # section 数据特征缩放和数据加噪

    start_time = time.time()  # 开始时间
    # 对不同维度进行标准化
    X = StandardScaler().fit_transform(X)
    # 记录原始索引
    original_indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.3, random_state=1)
    # 加入随机噪声的比例
    noise_level = 0.2
    # 计算噪声数量
    n_samples = X.shape[0]
    n_noise = int(noise_level * n_samples)
    # 随机选择要添加噪声的样本
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    # 添加高斯噪声到特征
    X_copy = np.copy(X)
    # X_copy[noise_indices] += np.random.normal(0, 1, (n_noise, X.shape[1]))
    # 从加噪数据中生成加噪训练数据和加噪测试数据
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
    end_time = time.time()  # 结束时间
    print("数据特征缩放和数据加噪耗时：", end_time - start_time)

    # section 找到有影响力的特征 M𝑐 (𝑅, 𝐴, M)

    start_time = time.time()  # 开始时间
    # choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)
    import re

    i = len(feature_names)
    np.random.seed(1)
    categorical_names = {}
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_copy, y_train)

    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(data_copy.iloc[:, feature])
        data_copy.iloc[:, feature] = le.transform(data_copy.iloc[:, feature])
        categorical_names[feature] = le.classes_

    explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_name,
                                     categorical_features=categorical_features,
                                     categorical_names=categorical_names, kernel_width=3)

    predict_fn = lambda x: rf_model.predict_proba(x)
    exp = explainer.explain_instance(X_train[i], predict_fn, num_features=len(feature_names) // 2)
    # 获取最具影响力的特征及其权重
    top_features = exp.as_list()
    top_feature_names = [re.search(r'([a-zA-Z_]\w*)', feature[0]).group(0).strip() for feature in top_features]
    top_k_indices = [feature_names.index(name) for name in top_feature_names]
    print("LIME检验的最有影响力的属性的索引：{}".format(top_k_indices))

    # # 获取最具影响力的特征及其权重
    # top_features = exp.as_list()
    # important_features = []
    # for feature_set in top_features:
    #     feature_long = feature_set[0]
    #     for feature in feature_names:
    #         if set(feature).issubset(set(feature_long)):
    #             important_features.append(feature)
    #             break
    #
    # top_k_indices = [feature_names.index(feature_name) for feature_name in important_features]
    # print("LIME检验的最有影响力的属性的索引：{}".format(top_k_indices))
    end_time = time.time()  # 结束时间
    print("有影响力的特征计算耗时：", end_time - start_time)

    # SECTION random forest模型的实现

    start_time = time.time()  # 开始时间

    # subsection 原始数据集上训练的random forest模型在训练集和测试集中分错的样本比例

    print("*" * 100)
    rf_clf = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42, class_weight='balanced')
    rf_clf.fit(X_train, y_train)
    train_label_pred = rf_clf.predict(X_train)
    test_label_pred = rf_clf.predict(X_test)

    # 使用 np.unique 统计不同标签及其出现次数
    unique_labels, counts = np.unique(train_label_pred, return_counts=True)

    # 打印结果
    for label, count in zip(unique_labels, counts):
        print(f"干净训练集Label: {label}, 预测Count: {count}")

    unique_labels, counts = np.unique(test_label_pred, return_counts=True)

    # 打印结果
    for label, count in zip(unique_labels, counts):
        print(f"干净测试集Label: {label}, 预测Count: {count}")

    print("*" * 100)

    # 训练样本中被random forest模型错误分类的样本
    wrong_classified_train_indices = np.where(y_train != train_label_pred)[0]
    print("训练样本中被random forest模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices) / len(y_train))

    # 测试样本中被random forest模型错误分类的样本
    wrong_classified_test_indices = np.where(y_test != test_label_pred)[0]
    print("测试样本中被random forest模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices) / len(y_test))

    # 整体数据集D中被random forest模型错误分类的样本
    print("完整数据集D中被random forest模型错误分类的样本占总完整数据的比例：",
          (len(wrong_classified_train_indices) + len(wrong_classified_test_indices)) / (len(y_train) + len(y_test)))

    # subsection 加噪数据集上训练的random forest模型在训练集和测试集中分错的样本比例

    print("*" * 100)
    train_label_pred_noise = rf_model.predict(X_train_copy)
    test_label_pred_noise = rf_model.predict(X_test_copy)

    # 使用 np.unique 统计不同标签及其出现次数
    unique_labels, counts = np.unique(train_label_pred_noise, return_counts=True)

    # 打印结果
    for label, count in zip(unique_labels, counts):
        print(f"加噪训练集Label: {label}, 预测Count: {count}")

    unique_labels, counts = np.unique(test_label_pred_noise, return_counts=True)

    # 打印结果
    for label, count in zip(unique_labels, counts):
        print(f"加噪测试集Label: {label}, 预测Count: {count}")

    print("*" * 100)

    # 加噪训练样本中被random forest模型错误分类的样本
    wrong_classified_train_indices_noise = np.where(y_train != train_label_pred_noise)[0]
    train_wrong_ratio = len(wrong_classified_train_indices_noise) / len(y_train)
    print("加噪训练样本中被random forest模型错误分类的样本占总加噪训练样本的比例：", len(wrong_classified_train_indices_noise) / len(y_train))

    # 加噪测试样本中被random forest模型错误分类的样本
    wrong_classified_test_indices_noise = np.where(y_test != test_label_pred_noise)[0]
    print("加噪测试样本中被random forest模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices_noise) / len(y_test))

    # 整体加噪数据集D中被random forest模型错误分类的样本
    print("完整数据集D中被random forest模型错误分类的样本占总完整数据的比例：",
          (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise)) / (len(y_train) + len(y_test)))

    # subsection 用多种指标评价加噪数据集中random forest的预测效果

    """Precision/Recall/F1指标"""
    print("*" * 100)

    # average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
    # average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
    # average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
    # average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。
    y_test_pred = test_label_pred_noise
    print("random forest模型在加噪测试集中的分类精确度：" + str(precision_score(y_test, y_test_pred, average='weighted')))
    print("random forest模型在加噪测试集中的分类召回率：" + str(recall_score(y_test, y_test_pred, average='weighted')))
    print("random forest模型在加噪测试集中的分类F1分数：" + str(f1_score(y_test, y_test_pred, average='weighted')))

    """ROC-AUC指标"""
    # y_test_prob = rf_model.predict_proba(X_test_copy)
    # # 对于二分类任务
    # roc_auc_test = roc_auc_score(y_test, y_test_prob[:, 1])  # 使用第二类的概率
    # print("random forest模型在加噪测试集中的ROC-AUC分数：" + str(roc_auc_test))

    """PR AUC指标(不支持多分类)"""
    # # 计算预测概率
    # y_scores = rf_model_noise.predict_proba(X_test)
    # # 遍历每个类别
    # pr_scores = []
    # for i in range(y_scores.shape[1]):
    #     precision, recall, _ = precision_recall_curve(y_test, y_scores[:, i])
    #     pr_auc = auc(recall, precision)
    #     pr_scores.append(pr_auc)
    #     print(f"random forest模型在修复测试集中的PR AUC 分数（类 {i}）: {pr_auc}")
    # # 如果需要计算所有类的宏平均 PR 分数
    # macro_pr_score = sum(pr_scores) / len(pr_scores)
    # print("random forest模型在修复测试集中的宏平均AP分数:", macro_pr_score)

    """AP指标(不支持多分类)"""
    # # 计算预测概率
    # y_scores = rf_model_noise.predict_proba(X_test)
    # # 计算每个类别的 Average Precision
    # ap_scores = []
    # for i in range(y_scores.shape[1]):
    #     ap_score = average_precision_score(y_test, y_scores[:, i])
    #     ap_scores.append(ap_score)
    #     print(f"random forest模型在修复测试集中的AP分数（类 {i}）: {ap_score}")
    #
    # # 如果需要计算所有类的宏平均 AP 分数
    # macro_ap_score = sum(ap_scores) / len(ap_scores)
    # print("random forest模型在修复测试集中的宏平均AP分数:", macro_ap_score)

    end_time = time.time()  # 结束时间
    print("random forest模型训练和预测耗时：", end_time - start_time)

    # section 找到loss(M, D, 𝑡) > 𝜆的元组

    start_time = time.time()  # 开始时间

    # choice 使用sklearn库中的hinge损失函数
    # decision_values = rf_model.decision_function(X_copy)
    # predicted_labels = np.argmax(decision_values, axis=1)
    # # 计算每个样本的hinge损失
    # num_samples = X_copy.shape[0]
    # num_classes = rf_model.classes_.shape[0]
    # hinge_losses = np.zeros((num_samples, num_classes))
    # hinge_loss = np.zeros(num_samples)
    # for i in range(num_samples):
    #     correct_class = int(y[i])
    #     for j in range(num_classes):
    #         if j != correct_class:
    #             loss_j = max(0, 1 - decision_values[i, correct_class] + decision_values[i, j])
    #             hinge_losses[i, j] = loss_j
    #     hinge_loss[i] = np.max(hinge_losses[i])
    #
    # # 在所有加噪数据D中损失函数高于阈值的样本索引
    # ugly_outlier_candidates = np.where(hinge_loss > 1)[0]
    # # print("D中损失函数高于损失阈值的样本索引为：", ugly_outlier_candidates)

    # choice 使用交叉熵损失函数
    # # 获取决策值
    # decision_values = rf_model.decision_function(X_copy)
    # # 将决策值转换为适用于 Softmax 的二维数组
    # decision_values_reshaped = decision_values.reshape(-1, 1)  # 变成 (n_samples, 1)
    # # 应用 Softmax 函数（可以手动实现或使用 scipy）
    # y_pred = softmax(np.hstack((decision_values_reshaped, -decision_values_reshaped)), axis=1)
    # # 创建 OneHotEncoder 实例
    # encoder = OneHotEncoder(sparse=False)
    # # 预测y_test的值，并与y_train组合成为y_ground
    # y_test_pred = rf_model.predict(X_test_copy)
    # y_ground = np.hstack((y_train, y_test_pred))
    # # 对y_ground进行独热编码
    # y_true = encoder.fit_transform(y_ground.reshape(-1, 1))
    # # 计算每个样本的损失
    # loss_per_sample = -np.sum(y_true * np.log(y_pred + 1e-12), axis=1)
    # # 计算测试集平均多分类交叉熵损失
    # average_loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
    # bad_samples = np.where(loss_per_sample > average_loss)[0]
    # good_samples = np.where(loss_per_sample <= average_loss)[0]
    # # 在所有加噪数据D中损失函数高于阈值的样本索引
    # # ugly_outlier_candidates = np.where(bad_samples > 1)[0]
    # ugly_outlier_candidates = bad_samples
    #
    # # bad_num_threshold = int(train_wrong_ratio * X_copy.shape[0])
    # # # 计算每个损失值与0的绝对差值
    # # abs_loss = np.abs(loss_per_sample)
    # # # 找到最接近0的样本的索引
    # # # closest_indices = np.argsort(abs_loss)[:bad_num_threshold]
    # # closest_indices = np.argsort(abs_loss)[:bad_num_threshold][::-1]

    # choice 判定分类错误的样本
    y_pred = rf_model.predict(X_copy)
    ugly_outlier_candidates = np.where(y_pred != y)[0]
    # 提取对应索引的标签
    selected_labels = y[ugly_outlier_candidates]
    print("ugly_outlier_candidates的数量：", len(ugly_outlier_candidates))
    print("ugly_outlier_candidates中标签为1的样本数量：", np.sum(selected_labels == 1))
    end_time = time.time()  # 结束时间
    print("损失函数实现耗时：", end_time - start_time)

    # section 谓词outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )的实现，找到所有有影响力的特征下的异常元组

    start_time = time.time()  # 开始时间

    outlier_feature_indices = {}
    threshold = 0.01
    for column_indice in top_k_indices:
        select_feature = feature_names[column_indice]
        select_column_data = data_copy[select_feature].values
        max_value = np.max(select_column_data)
        min_value = np.min(select_column_data)
        sorted_indices = np.argsort(select_column_data)
        sorted_data = select_column_data[sorted_indices]
        # 找到A属性下的所有异常值
        outliers = []
        outliers_index = []
        # 检查列表首尾元素
        if len(sorted_data) > 1:
            if (sorted_data[1] - sorted_data[0] >= threshold):
                outliers.append(sorted_data[0])
                outliers_index.append(sorted_indices[0])
            if (sorted_data[-1] - sorted_data[-2] >= threshold):
                outliers.append(sorted_data[-1])
                outliers_index.append(sorted_indices[-1])
        # 检查中间元素
        for i in range(1, len(sorted_data) - 1):
            current_value = sorted_data[i]
            left_value = sorted_data[i - 1]
            right_value = sorted_data[i + 1]
            if (current_value - left_value >= threshold) and (right_value - current_value >= threshold):
                outliers.append(current_value)
                outliers_index.append(sorted_indices[i])
        outliers_index_numpy = np.array(outliers_index)
        intersection = np.intersect1d(np.array(outliers_index), ugly_outlier_candidates)
        # print("有影响力的特征A下同时满足outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )和loss(M, D, 𝑡) > 𝜆的所有异常值索引为：", intersection)
        outlier_feature_indices[column_indice] = intersection
    # print(outlier_feature_indices)

    end_time = time.time()  # 结束时间
    print("outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )聚合函数实现耗时：", end_time - start_time)

    # section 确定数据中需要修复的元组

    start_time = time.time()  # 开始时间

    outlier_tuple_set = set()
    for value in outlier_feature_indices.values():
        outlier_tuple_set.update(value)
    X_copy_repair_indices = list(outlier_tuple_set)
    X_copy_repair_indices.extend(ugly_outlier_candidates)
    X_copy_repair = X_copy[X_copy_repair_indices]
    print("总的样本数量为：", len(X_copy))
    print("需要修复的样本数量为：", len(X_copy_repair_indices))
    y_repair = y[X_copy_repair_indices]

    # 生成保留的行索引
    rows_to_keep = np.setdiff1d(np.arange(X_copy.shape[0]), X_copy_repair_indices)

    # 使用保留的行索引选择D'中的正常数据
    # 无需修复的特征和标签值
    X_copy_inners = X_copy[rows_to_keep]
    y_inners = y[rows_to_keep]

    end_time = time.time()  # 结束时间
    print("确定ugly outliers耗时：", end_time - start_time)

    # # section 方案一：对X_copy中需要修复的元组进行标签修复（knn方法）
    # #  需要修复的元组通过异常值检测器检测到的元组和random forest分类错误的元组共同确定（取并集）
    #
    # # subsection 尝试修复异常数据的标签
    #
    # knn = KNeighborsClassifier(n_neighbors=3)
    # knn.fit(X_copy_inners, y_inners)
    #
    # # 统计y_inners不同值及其数量
    # unique_values, counts = np.unique(y_inners, return_counts=True)
    #
    # # 输出结果
    # for value, count in zip(unique_values, counts):
    #     print(f"标签: {value}, 数量: {count}")
    #
    # # 预测异常值
    # y_pred = knn.predict(X_copy_repair)
    #
    # # 替换异常值
    # y[X_copy_repair_indices] = y_pred
    # y_train = y[train_indices]
    # y_test = y[test_indices]
    #
    # # subsection 重新在修复后的数据上训练random forest模型
    #
    # print("*"*100)
    # rf_repair = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42, class_weight='balanced')
    # rf_repair.fit(X_train_copy, y_train)
    # y_train_pred = rf_repair.predict(X_train_copy)
    # y_test_pred = rf_repair.predict(X_test_copy)
    #
    # # 使用 np.unique 统计不同标签及其出现次数
    # unique_labels, counts = np.unique(y_train_pred, return_counts=True)
    #
    # # 打印结果
    # for label, count in zip(unique_labels, counts):
    #     print(f"修复训练集Label: {label}, 预测Count: {count}")
    #
    # unique_labels, counts = np.unique(y_test_pred, return_counts=True)
    #
    # # 打印结果
    # for label, count in zip(unique_labels, counts):
    #     print(f"修复测试集Label: {label}, 预测Count: {count}")
    #
    # print("*" * 100)
    # # 训练样本中被random forest模型错误分类的样本
    # wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
    # print("加噪标签修复后，训练样本中被random forest模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))
    #
    # # 测试样本中被random forest模型错误分类的样本
    # wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
    # print("加噪标签修复后，测试样本中被random forest模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))
    #
    # # 整体数据集D中被random forest模型错误分类的样本
    # print("加噪标签修复后，完整数据集D中被random forest模型错误分类的样本占总完整数据的比例：",
    #       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))
    #
    # # subsection 用多种指标评价random forest在修复后的数据上的预测效果
    #
    # """Precision/Recall/F1指标"""
    # print("*" * 100)
    #
    # # average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
    # # average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
    # # average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
    # # average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。
    #
    # print("random forest模型在修复测试集中的分类精确度：" + str(precision_score(y_test, y_test_pred, average='weighted')))
    # print("random forest模型在修复测试集中的分类召回率：" + str(recall_score(y_test, y_test_pred, average='weighted')))
    # print("random forest模型在修复测试集中的分类F1分数：" + str(f1_score(y_test, y_test_pred, average='weighted')))
    #
    # """ROC-AUC指标"""
    # # y_test_prob = rf_repair.predict_proba(X_test_copy)
    # # roc_auc_test = roc_auc_score(y_test, y_test_prob[:, 1])  # 一对多方式
    # # print("random forest模型在修复测试集中的ROC-AUC分数：" + str(roc_auc_test))
    #
    # """PR AUC指标(不支持多分类)"""
    # # # 计算预测概率
    # # y_scores = rf_repair.predict_proba(X_test)
    # # # 计算 Precision 和 Recall
    # # precision, recall, _ = precision_recall_curve(y_test, y_scores)
    # # # 计算 PR AUC
    # # pr_auc = auc(recall, precision)
    # # print("random forest模型在修复测试集中的PR AUC 分数:", pr_auc)
    # #
    # """AP指标(不支持多分类)"""
    # # # 计算预测概率
    # # y_scores = rf_repair.predict_proba(X_test)
    # # # 计算 Average Precision
    # # ap_score = average_precision_score(y_test, y_scores)
    # # print("random forest模型在修复测试集中的AP分数:", ap_score)

    # # section 方案二：对X_copy中需要修复的元组进行特征修复（统计方法修复）
    # #  需要修复的元组通过异常值检测器检测到的元组和random forest分类错误的元组共同确定（取并集）(修复效果由于监督/无监督基准)
    #
    # # subsection 确定有影响力特征中的离群值并采用均值修复
    # for i in range(X_copy.shape[1]):
    #     if i in top_k_indices:
    #         column_data = X_copy[:, i]
    #         mean = np.mean(column_data)
    #         # 将所有需要修复的行对应的列位置的元素替换为均值
    #         intersection = X_copy_repair_indices
    #         X_copy[intersection, i] = mean
    #
    # X_train_copy = X_copy[train_indices]
    # X_test_copy = X_copy[test_indices]
    #
    # # subsection 重新在修复后的数据上训练random forest模型
    #
    # rf_repair = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42, class_weight='balanced')
    # rf_repair.fit(X_train_copy, y_train)
    # y_train_pred = rf_repair.predict(X_train_copy)
    # y_test_pred = rf_repair.predict(X_test_copy)
    #
    # print("*" * 100)
    # # 训练样本中被random forest模型错误分类的样本
    # wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
    # print("加噪标签修复后，训练样本中被random forest模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))
    #
    # # 测试样本中被random forest模型错误分类的样本
    # wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
    # print("加噪标签修复后，测试样本中被random forest模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))
    #
    # # 整体数据集D中被random forest模型错误分类的样本
    # print("加噪标签修复后，完整数据集D中被random forest模型错误分类的样本占总完整数据的比例：",
    #       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

    # # section 方案三：对X_copy中需要修复的元组借助knn进行修复，choice1 将异常元组中的元素直接设置为nan(修复误差太大，修复后准确性下降)
    # #  choice2 仅将有影响力特征上的元素设置为np.nan
    #
    # # # choice 将异常元组中的所有元素设置为nan
    # # for i in range(X_copy.shape[1]):
    # #     X_copy[X_copy_repair_indices, i] = np.nan
    #
    # # choice 仅将异常元组中的有影响力的元素设置为nan
    # for i in top_k_indices:
    #     X_copy[X_copy_repair_indices, i] = np.nan
    #
    # # choice 使用knn修复所有被标记为nan的异常特征
    # # 创建 KNN Imputer 对象
    # knn_imputer = KNNImputer(n_neighbors=5)
    #
    # # 使用 KNN 算法填补异常特征
    # X_copy = knn_imputer.fit_transform(X_copy)
    # X_train_copy = X_copy[train_indices]
    # X_test_copy = X_copy[test_indices]
    #
    # rf_repair = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42, class_weight='balanced')
    # rf_repair.fit(X_train_copy, y_train)
    # y_train_pred = rf_repair.predict(X_train_copy)
    # y_test_pred = rf_repair.predict(X_test_copy)
    #
    # print("*" * 100)
    # # 训练样本中被random forest模型错误分类的样本
    # wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
    # print("借助knn修复需要修复的样本后，训练样本中被random forest模型错误分类的样本占总训练样本的比例：",
    #       len(wrong_classified_train_indices)/len(y_train))
    #
    # # 测试样本中被random forest模型错误分类的样本
    # wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
    # print("借助knn修复需要修复的样本后，测试样本中被random forest模型错误分类的样本占总测试样本的比例：",
    #       len(wrong_classified_test_indices)/len(y_test))
    #
    # # 整体数据集D中被random forest模型错误分类的样本
    # print("借助knn修复需要修复的样本后，完整数据集D中被random forest模型错误分类的样本占总完整数据的比例：",
    #       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))
    #       /(len(y_train) + len(y_test)))

    # # section 方案四：将X_copy中训练集和测试集需要修复的元组直接删除，在去除后的训练集上训练random forest模型
    #
    # set_X_copy_repair = set(X_copy_repair_indices)
    #
    # # 计算差集，去除训练集中需要修复的的元素
    # set_train_indices = set(train_indices)
    # remaining_train_indices = list(set_train_indices - set_X_copy_repair)
    # X_train_copy_repair = X_copy[remaining_train_indices]
    # y_train_copy_repair = y[remaining_train_indices]
    #
    # # # choice 计算差集，去除测试集中需要修复的的元素
    # # set_test_indices = set(test_indices)
    # # remaining_test_indices = list(set_test_indices - set_X_copy_repair)
    # # X_test_copy_repair = X_copy[remaining_test_indices]
    # # y_test_copy_repair = y[remaining_test_indices]
    #
    # # choice 不删除测试集中的离群样本
    # X_test_copy_repair = X_copy[test_indices]
    # y_test_copy_repair = y[test_indices]
    #
    # # subsection 重新在修复后的数据上训练random forest模型
    #
    # rf_repair = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42, class_weight='balanced')
    # rf_repair.fit(X_train_copy_repair, y_train_copy_repair)
    # y_train_pred = rf_repair.predict(X_train_copy_repair)
    # y_test_pred = rf_repair.predict(X_test_copy_repair)
    #
    # print("*" * 100)
    # # 训练样本中被random forest模型错误分类的样本
    # wrong_classified_train_indices = np.where(y_train_copy_repair != y_train_pred)[0]
    # print("删除需要修复的样本后，训练样本中被random forest模型错误分类的样本占总训练样本的比例：",
    #       len(wrong_classified_train_indices)/len(y_train_copy_repair))
    #
    # # 测试样本中被random forest模型错误分类的样本
    # wrong_classified_test_indices = np.where(y_test_copy_repair != y_test_pred)[0]
    # print("删除需要修复的样本后，测试样本中被random forest模型错误分类的样本占总测试样本的比例：",
    #       len(wrong_classified_test_indices)/len(y_test_copy_repair))
    #
    # # 整体数据集D中被random forest模型错误分类的样本
    # print("删除需要修复的样本后，完整数据集D中被random forest模型错误分类的样本占总完整数据的比例：",
    #       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))
    #       /(len(y_train_copy_repair) + len(y_test_copy_repair)))

    # # section 方案五：训练机器学习模型（随机森林模型），修复标签值
    #
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.metrics import mean_absolute_error
    #
    # # subsection 修复标签值
    # # 训练模型
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_copy_inners, y_inners)  # 使用正常样本训练模型
    #
    # # 预测离群样本的标签
    # y_repair_pred = model.predict(X_copy_repair)
    #
    # # 计算预测的准确性（可选）
    # mae = mean_absolute_error(y_repair, y_repair_pred)
    # print(f'Mean Absolute Error: {mae}')
    #
    # # subsection 修复特征值
    #
    # X_copy[X_copy_repair_indices] = X_copy_repair
    # y[X_copy_repair_indices] = y_repair_pred
    # X_train_copy = X_copy[train_indices]
    # X_test_copy = X_copy[test_indices]
    # y_train = y[train_indices]
    # y_test = y[test_indices]
    #
    # # subsection 重新在修复后的数据上训练random forest模型
    #
    # rf_repair = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42, class_weight='balanced')
    # rf_repair.fit(X_train_copy, y_train)
    # y_train_pred = rf_repair.predict(X_train_copy)
    # y_test_pred = rf_repair.predict(X_test_copy)
    #
    # print("*" * 100)
    # # 训练样本中被random forest模型错误分类的样本
    # wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
    # print("加噪标签修复后，训练样本中被random forest模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))
    #
    # # 测试样本中被random forest模型错误分类的样本
    # wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
    # print("加噪标签修复后，测试样本中被random forest模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))
    #
    # # 整体数据集D中被random forest模型错误分类的样本
    # print("加噪标签修复后，完整数据集D中被random forest模型错误分类的样本占总完整数据的比例：",
    #       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

    # # section 方案六：训练机器学习模型(随机森林模型)，修复特征值（修复时间很久，慎用）
    # #  依次将有影响力的特征作为要修复的标签（连续特征对应回归模型，分类特征对应分类模型），使用其他特征参与训练
    #
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.metrics import mean_absolute_error
    #
    # # subsection 修复特征值
    #
    # for i in top_k_indices:
    #     y_train_inf = X_copy_inners[:, i]
    #     columns_to_keep = np.delete(range(X_copy_inners.shape[1]), i)
    #     X_train_remain = X_copy_inners[:, columns_to_keep]
    #     if i in categorical_features:
    #         model = RandomForestClassifier(n_estimators=100, random_state=42)
    #     else:
    #         model = RandomForestRegressor(n_estimators=100, random_state=42)
    #     model.fit(X_train_remain, y_train_inf)  # 使用正常样本训练模型
    #     X_test_repair = X_copy_repair[:, columns_to_keep]
    #     y_test_pred = model.predict(X_test_repair)
    #     X_copy_repair[:, i] = y_test_pred
    #
    # X_copy[X_copy_repair_indices] = X_copy_repair
    # X_train_copy = X_copy[train_indices]
    # X_test_copy = X_copy[test_indices]
    # y_train = y[train_indices]
    # y_test = y[test_indices]
    #
    # # subsection 重新在修复后的数据上训练random forest模型
    #
    # rf_repair = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42, class_weight='balanced')
    # rf_repair.fit(X_train_copy, y_train)
    # y_train_pred = rf_repair.predict(X_train_copy)
    # y_test_pred = rf_repair.predict(X_test_copy)
    #
    # print("*" * 100)
    # # 训练样本中被random forest模型错误分类的样本
    # wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
    # print("加噪标签修复后，训练样本中被random forest模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))
    #
    # # 测试样本中被random forest模型错误分类的样本
    # wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
    # print("加噪标签修复后，测试样本中被random forest模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))
    #
    # # 整体数据集D中被random forest模型错误分类的样本
    # print("加噪标签修复后，完整数据集D中被random forest模型错误分类的样本占总完整数据的比例：",
    #       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))