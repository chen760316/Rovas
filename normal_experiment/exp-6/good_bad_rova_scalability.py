"""
𝑅(𝑡) ∧outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃) ∧loss(M, D, 𝑡) > 𝜆 → good(𝑡)
𝑅(𝑡) ∧ outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃) ∧ loss(M, D, 𝑡) ≤ 𝜆 → bad(𝑡)
测试good/bad outliers检测规则的可扩展性（重点关注谓词的可扩展性）
"""

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
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
import re
import time
from memory_profiler import memory_usage
import tracemalloc
from contextlib import contextmanager

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

@contextmanager
def memory_timer():
    tracemalloc.start()  # 开始跟踪内存
    yield  # 允许执行代码块
    current, peak = tracemalloc.get_traced_memory()  # 获取当前和峰值内存
    tracemalloc.stop()  # 停止跟踪内存
    print(f"Current Memory Usage: {current / 10**6:.2f} MiB")
    print(f"Peak Memory Usage: {peak / 10**6:.2f} MiB")

with memory_timer():
    t0 = time.time()  # 开始时间
    epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_trans = 64
    random_state = 42

    # section 标准数据集处理

    start_time = time.time()  # 开始时间

    # choice drybean数据集(效果好)
    # file_path = "../datasets/multi_class_to_outlier/drybean_outlier.csv"
    # data = pd.read_csv(file_path)

    # choice obesity数据集(效果好)
    # file_path = "../datasets/multi_class_to_outlier/obesity_outlier.csv"
    # data = pd.read_csv(file_path)

    # choice balita数据集(SVM拟合效果差，但修复后效果提升显著)
    # file_path = "../datasets/multi_class_to_outlier/balita_outlier.csv"
    # data = pd.read_csv(file_path)

    # choice apple数据集(效果提升小)
    # file_path = "../datasets/multi_class_to_outlier/apple_outlier.csv"
    # data = pd.read_csv(file_path)

    # choice adult数据集(效果提升明显)
    # file_path = "../datasets/multi_class_to_outlier/adult_outlier.csv"
    # data = pd.read_csv(file_path)

    # choice 真实异常检测数据集（本身不包含错误数据，不适合用于修复任务，且需要搭配非线性SVM）
    # file_path = "../datasets/real_outlier/Cardiotocography.csv"
    # file_path = "../datasets/real_outlier/annthyroid.csv"
    file_path = "../datasets/real_outlier/optdigits.csv"
    # file_path = "../datasets/real_outlier/PageBlocks.csv"
    # file_path = "../datasets/real_outlier/pendigits.csv"
    # file_path = "../datasets/real_outlier/satellite.csv"
    # file_path = "../datasets/real_outlier/shuttle.csv"
    # file_path = "../datasets/real_outlier/yeast.csv"
    data = pd.read_csv(file_path)

    # subsection 进行行采样和列采样
    print("原始数据集行数：", data.shape[0])
    print("原始数据集列数：", data.shape[1])
    # 随机采样固定比例的行
    sample_size = 0.5  # 行采样比例
    data = data.sample(frac=sample_size, random_state=1)

    # 随机采样固定比例的列
    sample_ratio = 0.5  # 列采样比例

    # 计算采样的列数（不包括标签列）
    num_features = data.shape[1] - 1  # 不包括标签列
    num_sampled_features = int(num_features * sample_ratio)

    # 随机选择特征列
    sampled_columns = data.columns[:-1].to_series().sample(n=num_sampled_features, random_state=42)

    # 提取采样的特征列和标签列
    label_name = data.columns[-1]
    data = data[sampled_columns.tolist() + [label_name]]

    print("采样后的数据集行数：", data.shape[0])
    print("采样后的数据集列数：", data.shape[1])

    # # 如果数据量超过20000行，就随机采样到20000行
    # if len(data) > 20000:
    #     data = data.sample(n=20000, random_state=42)

    enc = LabelEncoder()

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

    # SECTION M𝑜 (𝑡, D),针对元组异常的无监督异常检测器

    start_time = time.time()  # 开始时间
    clf = GOAD(epochs=epochs, device=device, n_trans=n_trans)
    clf.fit(X_train, y=None)
    end_time = time.time()  # 结束时间
    print("异常检测器训练耗时：", end_time - start_time)

    # SECTION 借助异常检测器，在训练集上进行异常值检测

    # train_scores = clf.decision_function(X_train)
    # train_pred_labels, train_confidence = clf.predict(X_train, return_confidence=True)
    # print("训练集中异常值判定阈值为：", clf.threshold_)
    # train_outliers_index = []
    # print("训练集样本数：", len(X_train))
    # for i in range(len(X_train)):
    #     if train_pred_labels[i] == 1:
    #         train_outliers_index.append(i)
    # # 在训练数据上的异常值索引下标
    # print("训练集中异常值索引：", train_outliers_index)
    # print("训练集中的异常值数量：", len(train_outliers_index))

    # SECTION 借助异常检测器，在测试集上进行异常值检测

    start_time = time.time()  # 开始时间
    test_scores = clf.decision_function(X_test)
    test_pred_labels, test_confidence = clf.predict(X_test, return_confidence=True)
    print("测试集中异常值判定阈值为：", clf.threshold_)
    test_outliers_index = []
    print("训练集样本数：", len(X_test))
    for i in range(len(X_test)):
        if test_pred_labels[i] == 1:
            test_outliers_index.append(i)
    # 在训练数据上的异常值索引下标
    print("训练集中异常值索引：", test_outliers_index)
    print("训练集中的异常值数量：", len(test_outliers_index))
    end_time = time.time()  # 结束时间
    print("检测器预测耗时：", end_time - start_time)

    # SECTION M𝑐 (𝑅, 𝐴,M)，在训练集中引入有影响力的特征

    start_time = time.time()  # 开始时间
    # # SUBSECTION 借助方差判别有影响力的特征
    # top_k_var = 6
    # variances = np.var(X_train, axis=0)
    # top_k_indices_var = np.argsort(variances)[-top_k_var:][::-1]
    # print("方差最大的前{}个特征的索引：{}".format(top_k_var, top_k_indices_var))
    #
    # # SUBSECTION 借助pearson相关系数筛选重要特征(将特征列和标签求pearson相关系数不太科学)
    # top_k_pearson = 6
    # y_trans = y_train.reshape(-1)
    # pearson_matrix = np.corrcoef(X_train.T, y_trans)
    # correlations = np.abs(pearson_matrix[0, 1:])
    # top_k_indices_pearson = np.argsort(correlations)[::-1][:top_k_pearson]
    # print("与标签y的Pearson相关系数绝对值最大的前{}个特征的索引：{}".format(top_k_pearson, top_k_indices_pearson))
    #
    # # SUBSECTION 借助互信息筛选重要特征(单个特征和标签之间的互信息)
    # top_k_mi = 6
    # y_trans_mi = y_train.reshape(-1)
    # mi = mutual_info_regression(X_train, y_trans_mi)
    # top_k_indices = np.argsort(mi)[::-1][:top_k_mi]
    # print("互信息最多的前{}个特征的索引：{}".format(top_k_mi, top_k_indices))
    #
    # # SUBSECTION 借助lasso筛选重要特征(特征的联合分布和标签之间的线性相关性)
    # alpha = 0.0001
    # top_k_lasso = 6
    # lasso = Lasso(alpha, max_iter=10000, tol=0.01)
    # lasso.fit(X_train, y_train)
    # coef = lasso.coef_
    # coef_abs = abs(coef)
    # top_k_indices = np.argsort(coef_abs)[::-1][:top_k_lasso]
    # print("lasso绝对值最大的前{}个属性的索引：{}".format(top_k_lasso, top_k_indices))
    #
    # SUBSECTION sklearn库的SelectKBest选择器，借助Fisher检验筛选最有影响力的k个特征
    # top_k_fisher = 6
    # selector = SelectKBest(score_func=f_classif, k=top_k_fisher)
    # y_trans_fisher = y_train.reshape(-1)
    # X_new = selector.fit_transform(X_train, y_trans_fisher)
    # # 获取被选中的特征的索引
    # selected_feature_indices = selector.get_support(indices=True)
    # print("SelectKBest选择器借助Fisher检验的最有影响力的{}个特征的索引：{}".format(top_k_fisher, selected_feature_indices))

    # # SUBSECTION 借助CART决策树筛选最有影响力的k个特征
    # top_k_cart = 6
    # classifier = DecisionTreeClassifier()
    # classifier.fit(X_train, y_train)
    # # 获取特征重要性得分
    # feature_importance = classifier.feature_importances_
    # # 根据重要性得分降序排序
    # sorted_indices = np.argsort(feature_importance)[::-1]
    # # 根据重要性得分降序排序
    # top_k_features = sorted_indices[:top_k_cart]
    # print("CART决策树检验的最有影响力的的前{}个属性的索引：{}".format(top_k_cart, top_k_features))
    #
    # # SUBSECTION sklearn库SelectFromModel选择器,它可以与任何具有coef_ 或 feature_importances_ 属性（如随机森林和决策树模型）的评估器一起使用来选择特征
    # classifier = RandomForestClassifier()
    # classifier.fit(X_train, y_train)
    # # 使用SelectFromModel来选择重要特征
    # sfm = SelectFromModel(classifier, threshold='mean', prefit=True)
    # X_selected = sfm.transform(X_train)
    # # 获取选择的特征索引
    # selected_idx = sfm.get_support(indices=True)
    # # 打印选择的特征索引
    # print("SelectFromModel选择器选择的特征索引:", selected_idx)
    #
    # # SUBSECTION 借助wrapper(包装)方法生成特征子集
    # model = LinearRegression()
    # # 初始化 RFE 特征选择器，选择要保留的特征数量
    # rfe = RFE(model, n_features_to_select=6)
    # # 拟合 RFE 特征选择器
    # rfe.fit(X_train, y_train)
    # # 输出选择的特征
    # indices = np.where(rfe.support_)[0]
    # print("wrapper(包装)方法选择的特征:", indices)
    # # 输出特征排名
    # print("wrapper(包装)方法下的特征排名:", rfe.ranking_)
    #

    # SUBSECTION 基于XGBoost模型以及XGB的特征重要性(一般情况下XGBoost的特征重要性评估方法更具准确性和可解释性)
    # top_k_xgboost = 6
    # gbtree = XGBClassifier(n_estimators=2000, max_depth=4, learning_rate=0.05, n_jobs=8)
    # gbtree.set_params(eval_metric='auc', early_stopping_rounds=100)
    # X_train_df = pd.DataFrame(X_train, columns=feature_names[:16])
    # X_test_df = pd.DataFrame(X_test, columns=feature_names[:16])
    # gbtree.fit(X_train, y_train,eval_set=[(X_test, y_test)], verbose=100)
    # feature_importances = gbtree.feature_importances_
    # top_k_indices = np.argpartition(-feature_importances, top_k_xgboost)[:top_k_xgboost]
    # print("XGBoost检验的最有影响力的的前{}个属性的索引：{}".format(top_k_xgboost, top_k_indices))

    # choice 无模型(非参数)方法中的Permutation Feature Importance-slearn，需要借助XGBoost

    # # 功能：permutation_importance 通过在数据集中对特征进行随机打乱（permutation）并评估模型性能的变化，来衡量特征的重要性。特征的重要性是通过查看特征打乱后模型性能的下降程度来确定的。
    # # 适用性：这个方法不依赖于特定的模型，因此可以与任何 sklearn 兼容的模型一起使用，包括但不限于线性模型、树模型、集成模型等。
    # result = permutation_importance(gbtree, X_train, y_train, n_repeats=10,random_state=42)
    # permutation_importance = result.importances_mean
    # top_k_permutation = np.argpartition(-permutation_importance, top_k_xgboost)[:top_k_xgboost]
    # print("Permutation Feature Importance-slearn检验的最有影响力的的前{}个属性的索引：{}".format(top_k_xgboost, top_k_permutation))

    # choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)
    import re

    svm_model = svm.SVC(class_weight='balanced', probability=True)
    svm_model.fit(X_train, y_train)
    # 特征数取4或6
    i = len(feature_names)
    np.random.seed(1)
    categorical_names = {}
    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(data.iloc[:, feature])
        data.iloc[:, feature] = le.transform(data.iloc[:, feature])
        categorical_names[feature] = le.classes_
    explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_name,
                                     categorical_features=categorical_features,
                                     categorical_names=categorical_names, kernel_width=3)
    # predict_proba 方法用于分类任务，predict 方法用于回归任务
    predict_fn = lambda x: svm_model.predict_proba(x)
    exp = explainer.explain_instance(X_train[i], predict_fn, num_features=len(feature_names) // 2)
    # 获取最具影响力的特征及其权重
    top_features = exp.as_list()
    top_feature_names = [re.search(r'([a-zA-Z_]\w*)', feature[0]).group(0).strip() for feature in top_features]
    top_k_indices = [feature_names.index(name) for name in top_feature_names]
    print("LIME检验的最有影响力的属性的索引：{}".format(top_k_indices))
    end_time = time.time()  # 结束时间
    print("有影响力的特征计算耗时：", end_time - start_time)

    # section outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃)聚合函数，如果对于D中所有元组s，t.A与s.A至少相差一个因子𝜃，则谓词返回true，否则返回false

    start_time = time.time()  # 开始时间
    # # subsection 从字面意思实现outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃)
    # threshold = 0.1
    # col_indices = 3
    # row_indices = 10
    # select_feature = feature_names[col_indices]
    # # 获得所选列的数据
    # select_column_data = data_copy[select_feature].values
    # # 找到所选列的最大值和最小值
    # max_value = np.max(select_column_data)
    # min_value = np.min(select_column_data)
    # # 找到t.A对应的值
    # t_value = data_copy.iloc[row_indices, col_indices]
    # # 对数据进行排序
    # # sorted_data = np.sort(select_column_data)
    # sorted_indices = np.argsort(select_column_data)
    # sorted_data = select_column_data[sorted_indices]
    # # 找到最接近的比 t_value 大的值和比 t_value 小的值
    # greater_than_t_value = sorted_data[sorted_data > t_value]
    # less_than_t_value = sorted_data[sorted_data < t_value]
    # # 找到与t_value最接近的更大的值和更小的值
    # if greater_than_t_value.size > 0:
    #     closest_greater = greater_than_t_value[0]  # 最近的大于 t_value 的值
    # else:
    #     closest_greater = t_value
    # if less_than_t_value.size > 0:
    #     closest_less = less_than_t_value[-1]  # 最近的小于 t_value 的值
    # else:
    #     closest_less = t_value
    # # 判断t.A是否是异常值
    # if max_value == t_value:
    #     print("元组t在属性A上的投影是异常值吗:", t_value - closest_less > threshold)
    # elif min_value == t_value:
    #     print("元组t在属性A上的投影是异常值吗:", closest_greater - t_value > threshold)
    # else:
    #     print("元组t在属性A上的投影是异常值吗:", t_value - closest_less > threshold and t_value - closest_less > threshold)
    # # 找到A属性下的所有异常值
    # outliers = []
    # outliers_index = []
    # # 检查列表首尾元素
    # if len(sorted_data) > 1:
    #     if (sorted_data[1] - sorted_data[0] >= threshold):
    #         outliers.append(sorted_data[0])
    #         outliers_index.append(sorted_indices[0])
    #     if (sorted_data[-1] - sorted_data[-2] >= threshold):
    #         outliers.append(sorted_data[-1])
    #         outliers_index.append(sorted_indices[-1])
    # # 检查中间元素
    # for i in range(1, len(sorted_data) - 1):
    #     current_value = sorted_data[i]
    #     left_value = sorted_data[i - 1]
    #     right_value = sorted_data[i + 1]
    #     if (current_value - left_value >= threshold) and (right_value - current_value >= threshold):
    #         outliers.append(current_value)
    #         outliers_index.append(sorted_indices[i])
    # # 在所有数据D下的异常值索引
    # print("A属性下所有异常值的索引为：", outliers_index)
    # print("A属性下所有异常值为：", outliers)

    # # subsection 采用间隔方法，使用Modified Z-score方法寻找满足outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃)的元组索引
    # def modified_z_score(points, thresh=3.5):
    #     if len(points.shape) == 1:
    #         points = points[:, None]
    #     median = np.median(points, axis=0)
    #     diff = np.sum((points - median)**2, axis=-1)
    #     diff = np.sqrt(diff)
    #     med_abs_deviation = np.median(diff)
    #     modified_z_score = 0.6745 * diff / med_abs_deviation
    #     return modified_z_score > thresh
    # col_indices = 3
    # select_feature = feature_names[col_indices]
    # feature_values = data_copy[select_feature].values
    # value_labels = modified_z_score(feature_values)
    # true_indices = np.where(value_labels)[0]
    # # 在所有数据D下的异常值索引
    # print("modified_z_score方法找到的符合条件的元组索引为：", true_indices)
    # print("modified_z_score方法找到的符合条件的元组数：", len(true_indices))

    # # subsection 采用间隔方法，使用2MADe方法寻找满足outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃)的元组索引
    # def calculate_made(data):
    #     median = np.median(data)  # 计算中位数
    #     abs_deviation = np.abs(data - median)  # 计算每个数据点与中位数的绝对误差
    #     mad = np.median(abs_deviation)  # 计算绝对误差均值
    #     made = 1.843 * mad
    #     return median, made
    #
    # col_indices = 3
    # select_feature = feature_names[col_indices]
    # feature_values = data_copy[select_feature].values
    # median, made = calculate_made(feature_values)
    # lower_threshold = median - 2 * made
    # upper_threshold = median + 2 * made
    # made_indices = np.where((feature_values > upper_threshold) | (feature_values < lower_threshold))[0]
    # # 在所有数据D下的异常值索引
    # print("2MADe方法找到的符合条件的元组索引为：", made_indices)
    # print("2MADe方法找到的符合条件的元组数：", len(made_indices))

    # # subsection 采用1.5IQR下的箱线图方法寻找满足outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃)的元组索引
    # def calculate_iqr(data):
    #     sorted_data = np.sort(data)  # 将数据集按升序排列
    #     q1 = np.percentile(sorted_data, 25)  # 计算下四分位数
    #     q3 = np.percentile(sorted_data, 75)  # 计算上四分位数
    #     iqr = q3 - q1
    #     return q1, q3, iqr
    # col_indices = 3
    # select_feature = feature_names[col_indices]
    # feature_values = data_copy[select_feature].values
    # q1, q3, iqr = calculate_iqr(feature_values)
    # lower_bound = q1 - 1.5 * iqr
    # upper_bound = q3 + 1.5 * iqr
    # box_plot_indices = np.where((feature_values > upper_bound) | (feature_values < lower_bound))[0]
    # # 在所有数据D下的异常值索引
    # print("箱线图方法找到的符合条件的元组索引为：", box_plot_indices)
    # print("箱线图方法找到的符合条件的元组数：", len(box_plot_indices))

    # # subsection 采用标准差法寻找满足outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃)的元组索引
    # col_indices = 3
    # select_feature = feature_names[col_indices]
    # feature_values = data_copy[select_feature].values
    # mean = feature_values.mean()
    # std = feature_values.std()
    # upper_bound = mean + 3 * std
    # lower_bound = mean - 3 * std
    # std_indices = np.where((feature_values > upper_bound) | (feature_values < lower_bound))[0]
    # # 在所有数据D下的异常值索引
    # print("标准差法找到的符合条件的元组索引为：", std_indices)
    # print("标准差法找到的符合条件的元组数：", len(std_indices))

    # # subsection 采用dist拟合单列数据分布的方法寻找满足outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃)的元组索引，数据中可能存在多个分布，可以考虑用分段函数建模（相对于filter方法误差范围很大）
    # col_indices = 3
    # select_feature = feature_names[col_indices]
    # feature_values = data_copy[select_feature].values
    # dist = distfit(todf=True)
    # dist.fit_transform(feature_values)
    # # 获取最佳分布
    # best_distribution_name = dist.model['name']
    # best_distribution_params = dist.model['params']
    # # 根据最佳分布名称和参数构建对应的概率分布对象
    # best_distribution = getattr(stats, best_distribution_name)(*best_distribution_params)
    # # 计算每个样本点的概率密度
    # densities = best_distribution.pdf(feature_values)
    # # 定义一个阈值，例如低于这个阈值的点被视为异常点
    # threshold = 0.01
    # # 找到异常点
    # outliers_indices = np.where(densities < threshold)[0]
    # dist.plot()
    # plt.show()
    # # 在所有数据D下的异常值索引
    # print("位于dist拟合的数据分布外的异常点索引:", outliers_indices)
    # print("位于dist拟合的数据分布外的异常点数量:", len(outliers_indices))

    # # subsection 采用filter fitting拟合单列数据分布的方法寻找满足outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃)的元组索引
    # col_indices = 5
    # select_feature = feature_names[col_indices]
    # feature_values = data_copy[select_feature].values
    # f = Fitter(feature_values, distributions=['norm', 't', 'laplace'])
    # f.fit()
    # # 计算最佳分布和最佳参数
    # pattern = r'\[(.*?)\]'
    # best_dist_name_key = f.get_best(method='sumsquare_error').keys()
    # best_dist_name = key_string = ', '.join(str(key) for key in best_dist_name_key)
    # best_params = None
    # for dist_name, params in f.fitted_param.items():
    #     if dist_name == best_dist_name:
    #         best_params = params
    #         break
    # if best_params is None:
    #     raise ValueError(f"No parameters found for the best distribution '{best_dist_name}'")
    # # 构建对应的概率分布对象
    # best_dist = getattr(stats, best_dist_name)(*best_params)
    # # 计算每个样本点的概率密度
    # densities = best_dist.pdf(feature_values)
    # # 设定阈值找出概率密度低于阈值的样本点作为异常点
    # threshold = 0.01  # 举例设定阈值
    # outliers_indices = np.where(densities < threshold)[0]
    # # 在所有数据D下的异常值索引
    # print("位于filter fitting拟合的数据分布外的异常点索引:", outliers_indices)
    # print("位于filter fitting拟合的数据分布外的异常点数量:", len(outliers_indices))

    # subsection 基于定义，采用分箱方法实现outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃)聚合函数

    # outlier_feature_indices = {}
    # threshold = 0.01
    # for column_indice in top_k_indices:
    #     select_feature = feature_names[column_indice]
    #     select_column_data = data_copy[select_feature].values
    #     max_value = np.max(select_column_data)
    #     min_value = np.min(select_column_data)
    #     sorted_indices = np.argsort(select_column_data)
    #     sorted_data = select_column_data[sorted_indices]
    #     # 找到A属性下的所有异常值
    #     outliers = []
    #     outliers_index = []
    #     # 检查列表首尾元素
    #     if len(sorted_data) > 1:
    #         if (sorted_data[1] - sorted_data[0] >= threshold):
    #             outliers.append(sorted_data[0])
    #             outliers_index.append(sorted_indices[0])
    #         if (sorted_data[-1] - sorted_data[-2] >= threshold):
    #             outliers.append(sorted_data[-1])
    #             outliers_index.append(sorted_indices[-1])
    #     # 检查中间元素
    #     for i in range(1, len(sorted_data) - 1):
    #         current_value = sorted_data[i]
    #         left_value = sorted_data[i - 1]
    #         right_value = sorted_data[i + 1]
    #         if (current_value - left_value >= threshold) and (right_value - current_value >= threshold):
    #             outliers.append(current_value)
    #             outliers_index.append(sorted_indices[i])
    #     outliers_index_numpy = np.array(outliers_index)
    #     intersection = np.array(outliers_index)
    #     # print("有影响力的特征A下同时满足outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )和loss(M, D, 𝑡) > 𝜆的所有异常值索引为：", intersection)
    #     outlier_feature_indices[column_indice] = intersection
    # # print(outlier_feature_indices)

    end_time = time.time()  # 结束时间
    print("outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃)聚合函数实现耗时：", end_time - start_time)

    # section imbalance d(𝐷, 𝑅, 𝑡.𝐴, 𝛿)，如果D中按t.A分组的元组数量比其他组的计数小A值(至少小一个因子𝛿)，则返回true，否则返回false

    start_time = time.time()  # 开始时间
    # subsection 从字面意思的具体值出现频率判断是否不平衡,实现imbalanced(𝐷, 𝑅, 𝑡.𝐴, 𝛿)，基础版本(存在分组数量与t.A相差𝛿)
    # import balanace.imbalanced as imbalance
    # col_indices = 16
    # row_indices = 10
    # delta = 2
    # feature = feature_names[col_indices]
    # data_copy = pd.read_excel(file_path).astype(str)
    # imbalanced = imbalance.Imbalanced(data_copy, feature)
    # # 在所有数据D下的元组下标
    # ta = data_copy.iloc[row_indices, col_indices]
    # print("所选列是否不平衡：", imbalanced.enum_check(ta, delta))

    # subsection 从字面意思实现imbalanced(𝐷, 𝑅, 𝑡.𝐴, 𝛿)，基础版本(所有分组数量与t.A至少相差𝛿)

    # def check_delta(value_counts, ta_count, delta):
    #     for value, count in value_counts.items():
    #         if value != ta and abs(count - ta_count) < delta:
    #             return False
    #     return True
    # col_indices = 16
    # row_indices = 10
    # delta = 2
    # feature = feature_names[col_indices]
    # data_copy = pd.read_excel(file_path).astype(str)
    # # 在所有数据D下的元组下标
    # ta = data_copy.iloc[row_indices, col_indices]
    # # 获得所选列的数据
    # select_column_data = data_copy[feature].values
    # equal_size = len(data_copy[feature]) / len(set(data_copy[feature]))
    # delta_threshold = delta * equal_size
    # # 获取所有值的计数
    # value_counts = pd.Series(select_column_data).value_counts()
    # ta_count = value_counts.get(ta, 0)
    # # 将ta分组计数与组内其他值计数进行比较
    # print("所选列是否不平衡：", check_delta(value_counts, ta_count, delta_threshold))

    # subsection 从统计视角实现imbalanced(𝐷, 𝑅, 𝑡.𝐴, 𝛿)，改进版本，对列进行标准化和分箱，判断分箱中的元素数是否达到不平衡（存在两分箱对应计数的差值至少为𝛿）

    # from sklearn.preprocessing import MinMaxScaler
    # # 设置分箱中元组数相差阈值
    # delta = 0.05
    # # 设置分组的间隔
    # interval = 0.01
    # # 初始化MinMaxScaler
    # scaler = MinMaxScaler()
    # col_indices = 3
    # select_feature = feature_names[col_indices]
    # data_copy = pd.read_excel(file_path)
    # data_copy[data.columns] = scaler.fit_transform(data[data.columns])
    # # 对每列数据进行分组
    # bins = np.arange(0, 1.01, interval)  # 生成0-1之间100个间隔的数组
    # digitized = np.digitize(data_copy[select_feature], bins)
    # unique_bins, counts = np.unique(digitized, return_counts=True)
    # print(f"列 '{select_feature}' 占据了 {len(unique_bins)} 个间隔")
    # # 统计包含最大元素数和最小元素数的差值
    # max_elements = np.max(counts)
    # min_elements = np.min(counts)
    # difference = max_elements - min_elements
    # print(f"列 '{select_feature}' bins中包含最多的元组数和最少的元组数相差了 {difference}")
    # print("所选列是否不平衡：", difference/data_copy.shape[0] >= delta)

    # subsection 从统计视角实现imbalanced(𝐷, 𝑅, 𝑡.𝐴, 𝛿)，改进版本，对列进行标准化和分箱，判断分箱中的元素数是否达到不平衡（t.A与其他所有分箱对应计数的差值至少为𝛿）

    # from sklearn.preprocessing import MinMaxScaler
    # # 设置分箱中元组数相差阈值
    # delta = 0.05
    # # 设置分组的间隔
    # interval = 0.01
    # # 初始化MinMaxScaler
    # scaler = MinMaxScaler()
    # col_indices = 16
    # row_indices = 100
    # select_feature = feature_names[col_indices]
    # data_minmax = pd.read_excel(file_path)
    # data_minmax[data.columns] = scaler.fit_transform(data[data.columns])
    # ta = data_minmax.iloc[row_indices, col_indices]
    # # 对每列数据进行分组
    # bins = np.arange(0, 1.01, interval)  # 生成0-1之间100个间隔的数组
    # digitized = np.digitize(data_minmax[select_feature], bins)
    # # 统计每个区间的计数
    # unique_bins, counts = np.unique(digitized, return_counts=True)
    # # 找到 ta 所在的间隔
    # ta_bin = np.digitize([ta], bins)[0]
    # # 找到 ta 所在间隔的计数
    # ta_count = counts[unique_bins == ta_bin][0]
    # # 设置最小支持数差值
    # min_diff = delta * data_minmax.shape[0]
    # # 判断 ta 所在间隔的支持数是否与其他所有间隔的支持数相差至少 min_diff
    # def check_min_diff(counts, ta_count, min_diff):
    #     for count in counts:
    #         if abs(count - ta_count) < min_diff:
    #             return False
    #     return True
    # # 进行检查
    # result = check_min_diff(counts, ta_count, min_diff)
    # print(f"Value of ta: {ta}")
    # print(f"Count in ta's bin: {ta_count}")
    # print("Is ta's bin count different from other bins by at least delta * data_copy.shape[0]?", result)

    # subsection 从统计视角实现imbalanced(𝐷, 𝑅, 𝑡.𝐴, 𝛿)，不设置𝛿的无阈值方法

    from sklearn.preprocessing import MinMaxScaler

    # 设置分组的间隔
    interval = 0.01
    # 初始化MinMaxScaler
    scaler = MinMaxScaler()
    col_indices = 1
    row_indices = 100
    select_feature = feature_names[col_indices]
    data[data.columns] = scaler.fit_transform(data[data.columns])
    # 在所有数据D下的元组下标
    ta = data.iloc[row_indices, col_indices]
    # 对每列数据进行分组
    bins = np.arange(0, 1.01, interval)  # 生成0-1之间100个间隔的数组
    digitized = np.digitize(data[select_feature], bins)
    # 统计每个区间的计数
    unique_bins, counts = np.unique(digitized, return_counts=True)
    # 找到 ta 所在的间隔
    ta_bin = np.digitize([ta], bins)[0]
    # 找到 ta 所在间隔的计数
    ta_count = counts[unique_bins == ta_bin][0]
    # 设置最小支持数差值
    median, made = calculate_made(counts)
    lower_threshold = median - 2 * made
    upper_threshold = median + 2 * made
    if ta_count < lower_threshold or ta_count > upper_threshold:
        print("所选列A在所选元组t处是不平衡的")
    else:
        print("所选列A在所选元组t处是平衡的")

    end_time = time.time()  # 结束时间
    print("imbalance d(𝐷, 𝑅, 𝑡.𝐴, 𝛿)聚合函数实现耗时：", end_time - start_time)

    # SECTION SDomain(𝐷, 𝑅, 𝐴, 𝜎)，如果D的A属性的不同值数量小于界限𝜎，则返回true

    start_time = time.time()  # 开始时间
    # subsection 从字面意思A列的不同值数量是否明显小于给定的阈值𝜎
    # import balanace.sdomain as sd
    # col_indices = 16
    # # 设置每列不同元素数量要达到的最小阈值
    # sigma = 2
    # feature = feature_names[col_indices]
    # data_copy = pd.read_excel(file_path)
    # imbalanced = sd.SDomian(data_copy, feature)
    # print("所选列的活动域是否小于设置阈值：", imbalanced.enum_check(sigma))

    # subsection 改进版本，对列的值进行标准化后分箱判断某列对应分箱的数量是否小于所有列分箱数的均值（不合理）

    # from sklearn.preprocessing import MinMaxScaler
    # # 设置分组的间隔
    # interval = 0.01
    # # 初始化MinMaxScaler
    # scaler = MinMaxScaler()
    # col_indices = 3
    # select_feature = feature_names[col_indices]
    # data_copy = pd.read_excel(file_path)
    # data_copy[data.columns] = scaler.fit_transform(data[data.columns])
    # # 对每列数据进行分组
    # bins = np.arange(0, 1.01, interval)  # 生成0-1之间100个间隔的数组
    # # 统计每列数据占据了多少个间隔
    # total_bins = 0
    # selected_bins = 0
    # for column in data_copy.columns:
    #     digitized = np.digitize(data_copy[column], bins)
    #     unique_bins, counts = np.unique(digitized, return_counts=True)
    #     print(f"列 '{column}' 占据了 {len(unique_bins)} 个间隔")
    #     total_bins += len(unique_bins)
    #     if column == select_feature:
    #         selected_bins = len(unique_bins)
    # mean_bins = total_bins / len(data_copy.columns)
    # print("所选特征是否活动域很小：", selected_bins < mean_bins)

    # subsection 改进版本，对列的值进行标准化后分箱判断某列对应分箱的数量是否过小（不设置阈值𝛿，采用2MADe统计方法）

    # from sklearn.preprocessing import MinMaxScaler
    # # 设置分组的间隔
    # interval = 0.01
    # col_indices = 3
    # selected_bins = 0
    # columns_bins = {}
    # columns_bins_count = []
    # # 初始化MinMaxScaler
    # scaler = MinMaxScaler()
    # select_feature = feature_names[col_indices]
    # data_minmax = pd.read_csv(file_path)
    # data_minmax[data.columns] = scaler.fit_transform(data[data.columns])
    # # 对每列数据进行分组
    # bins = np.arange(0, 1.01, interval)  # 生成0-1之间100个间隔的数组
    # # 统计每列数据占据了多少个间隔
    # for column in data_minmax.columns:
    #     digitized = np.digitize(data_minmax[column], bins)
    #     unique_bins, counts = np.unique(digitized, return_counts=True)
    #     print(f"列 '{column}' 占据了 {len(unique_bins)} 个间隔")
    #     columns_bins[column] = len(unique_bins)
    #     columns_bins_count.append(len(unique_bins))
    #     if column == select_feature:
    #         selected_bins = len(unique_bins)
    # median, made = calculate_made(np.array(columns_bins_count))
    # lower_threshold = median - 2 * made
    # upper_threshold = median + 2 * made
    # if selected_bins < lower_threshold:
    #     print("所选列的活动域过小")
    # else:
    #     print("所选列的活动域正常")

    end_time = time.time()  # 结束时间
    print("SDomain(𝐷, 𝑅, 𝐴, 𝜎)聚合函数实现耗时：", end_time - start_time)

    # section 谓词loss(M, D, 𝑡)的实现（交叉熵损失函数），检测good/bad outliers

    start_time = time.time()  # 开始时间
    print("*" * 100)
    # 获取决策值
    decision_values = svm_model.decision_function(X_copy)
    # 将决策值转换为适用于 Softmax 的二维数组
    decision_values_reshaped = decision_values.reshape(-1, 1)  # 变成 (n_samples, 1)
    # 应用 Softmax 函数（可以手动实现或使用 scipy）
    y_pred = softmax(np.hstack((decision_values_reshaped, -decision_values_reshaped)), axis=1)
    # 创建 OneHotEncoder 实例
    encoder = OneHotEncoder(sparse=False)
    # 预测y_test的值，并与y_train组合成为y_ground
    y_test_pred = svm_model.predict(X_test_copy)
    y_ground = np.hstack((y_train, y_test_pred))
    # 对y_ground进行独热编码
    y_true = encoder.fit_transform(y_ground.reshape(-1, 1))
    # 计算每个样本的损失
    loss_per_sample = -np.sum(y_true * np.log(y_pred + 1e-12), axis=1)
    # 计算测试集平均多分类交叉熵损失
    average_loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
    bad_samples = np.where(loss_per_sample > average_loss)[0]
    good_samples = np.where(loss_per_sample <= average_loss)[0]
    # 测试样本中的bad outliers索引
    bad_outliers_index = np.intersect1d(test_outliers_index, bad_samples)
    print("检测出的outliers中bad outliers的数量：", len(bad_outliers_index))
    # 测试样本中的good outliers索引
    good_outliers_index = np.intersect1d(test_outliers_index, good_samples)
    print("检测出的outliers中good outliers的数量：", len(good_outliers_index))
    # good outliers中分错的比例
    good_wrong_indies = []
    for i in good_outliers_index:
        true_label = y_test[i]
        if true_label != test_pred_labels[i]:
            good_wrong_indies.append(i)
    print("good outliers中样本分错的比例：", len(good_wrong_indies) / len(good_outliers_index))
    # bad outliers中分错的比例
    bad_wrong_indies = []
    for i in bad_outliers_index:
        true_label = y_test[i]
        if true_label != test_pred_labels[i]:
            bad_wrong_indies.append(i)
    print("bad outliers中样本分错的比例：", len(bad_wrong_indies) / len(bad_outliers_index))

    end_time = time.time()  # 结束时间
    print("loss(M, D, 𝑡)损失函数实现耗时：", end_time - start_time)

    # 对训练数据进行处理后重新训练SVM模型

    # SECTION 原始数据中的svm分类准确度
    # print("*" * 100)
    # print("原始训练集中SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
    # print("原始测试集中SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
    # print("*" * 100)

    # SECTION 舍弃掉SVM训练数据中分类错误（hinge损失函数高于1）且被判定为异常值的样本，重新在处理后的训练数据上训练SVM
    # # 生成布尔索引，为要删除的行创建布尔值数组
    # mask = np.ones(len(X_train), dtype=bool)
    # mask[inter_anomalies] = False
    # # 使用布尔索引删除那些既被判定为异常值，又使得hinge损失高于1的样本
    # X_train_split = X_train[mask]
    # y_train_split = y_train[mask]
    # # 重新训练SVM模型
    # svm_model_split = svm.SVC(class_weight='balanced', probability=True)
    # svm_model_split.fit(X_train_split, y_train_split)
    # print("*" * 100)
    # print("去除异常值中分类错误的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_split, svm_model_split.predict(X_train_split))))
    # print("去除异常值中分类错误的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_split.predict(X_test))))
    # print("*" * 100)

    # SECTION 舍弃掉SVM训练数据中分类错误（hinge损失函数高于1）的样本，重新在处理后的训练数据上训练SVM
    # mask_h = np.ones(len(X_train), dtype=bool)
    # mask_h[anomalies] = False
    # X_train_h = X_train[mask_h]
    # y_train_h = y_train[mask_h]
    # svm_model_h = svm.SVC(class_weight='balanced', probability=True)
    # svm_model_h.fit(X_train_h, y_train_h)
    # print("*" * 100)
    # print("去除损失高于阈值的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_h, svm_model_h.predict(X_train_h))))
    # print("去除损失高于阈值的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_h.predict(X_test))))
    # print("*" * 100)

    # SECTION 舍弃掉SVM训练数据中被判定为异常值的样本，重新在处理后的训练数据上训练SVM
    # mask_o = np.ones(len(X_train), dtype=bool)
    # mask_o[train_outliers_index] = False
    # X_train_o = X_train[mask_o]
    # y_train_o = y_train[mask_o]
    # svm_model_o = svm.SVC(class_weight='balanced', probability=True)
    # svm_model_o.fit(X_train_o, y_train_o)
    # print("*" * 100)
    # print("去除判定为异常的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_o, svm_model_o.predict(X_train_o))))
    # print("去除判定为异常的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_o.predict(X_test))))
    # print("*" * 100)

    t1 = time.time()  # 开始时间
    print("规则执行总耗时：", t1-t0)