"""
使用SVM训练集求解hinge损失函数，因为测试集标签不可见
SVM和异常检测器使用相同的训练集和测试集
异常检测器直接输出在训练集上的异常值
"""
# unsupervised methods
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

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行
pd.set_option('display.width', None)        # 不限制显示宽度
np.set_printoptions(threshold=np.inf)

def one_hot_encode(y, num_classes):
    y = y.astype(int)
    return np.eye(num_classes)[y]

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42
# 指定要检测的标签列类别为target_class时，样本中出现的异常值
target_class = 0

# SECTION kaggle datasets上的数据预处理
# SUBSECTION dry_bean数据集
file_path = "../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
drybean = pd.read_excel(file_path)
enc = LabelEncoder()
drybean['Class'] = enc.fit_transform(drybean['Class'])
X = drybean.values[:,0:16]
y = drybean.values[:,16]
# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
class_names = enc.classes_
categorical_features = [0, 6]
categorical_names = {}
for feature in categorical_features:
    le = LabelEncoder()
    le.fit(drybean.iloc[:, feature])
    drybean.iloc[:, feature] = le.transform(drybean.iloc[:, feature])
    categorical_names[feature] = le.classes_

# SUBSECTION obesity数据集
# file_path = "../kaggle_datasets/Obesity_prediction/obesity_data.csv"
# label_col_name = "ObesityCategory"
# data = pd.read_csv(file_path)
# enc = LabelEncoder()
# data[label_col_name] = enc.fit_transform(data[label_col_name])
# categorical_features = [1]
# categorical_names = {}
# feature_names = data.columns[:-1].tolist()
# # 对字符串列进行数值映射
# for feature in categorical_features:
#     le = LabelEncoder()
#     le.fit(data.iloc[:, feature])
#     data.iloc[:, feature] = le.transform(data.iloc[:, feature])
#     categorical_names[feature] = le.classes_
# data[feature_names] = data[feature_names].astype(float)
# num_col = data.shape[1]
# X = data.values[:, 0:num_col-1]
# y = data.values[:, num_col-1]
# # 对不同维度进行标准化
# X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# SUBSECTION wine-quality数据集
# file_path = "../UCI_datasets/wine+quality/winequality-white.csv"
# label_col_name = "quality"
# data = pd.read_csv(file_path, sep=';')
# enc = LabelEncoder()
# data[label_col_name] = enc.fit_transform(data[label_col_name])
# feature_names = data.columns[:-1].tolist()
# data[feature_names] = data[feature_names].astype(float)
# data[label_col_name] = data[label_col_name].astype(int)
# num_col = data.shape[1]
# X = data.values[:, 0:num_col-1]
# y = data.values[:, num_col-1]
# # 对不同维度进行标准化
# X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# SECTION GOAD异常检测器
clf = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf.fit(X_train, y=None)

# SECTION DeepSAD异常检测器
# clf = DeepSAD(epochs=1, hidden_dims=20,
#                    device=device,
#                    random_state=42)
# anom_id = np.where(y_train == 1)[0]
# known_anom_id = np.random.choice(anom_id, 10, replace=False)
# y_semi = np.zeros_like(y_train, dtype=int)
# y_semi[known_anom_id] = 1
# clf.fit(X_train, y_semi)

# SECTION 借助异常检测器，在训练集上进行异常值检测
train_scores = clf.decision_function(X_train)
train_pred_labels, train_confidence = clf.predict(X_train, return_confidence=True)
print("训练集中异常值判定阈值为：", clf.threshold_)
train_outliers_index = []
print("训练集样本数：", len(X_train))
for i in range(len(X_train)):
    if train_pred_labels[i] == 1:
        train_outliers_index.append(i)
print("训练集中异常值索引：", train_outliers_index)
print("训练集中的异常值数量：", len(train_outliers_index))

# SECTION 在训练集中引入有影响力的特征
# SUBSECTION 借助方差判别有影响力的特征
top_k_var = 6
variances = np.var(X_train, axis=0)
top_k_indices_var = np.argsort(variances)[-top_k_var:][::-1]
print("方差最大的前{}个特征的索引：{}".format(top_k_var, top_k_indices_var))

# SUBSECTION 借助pearson相关系数筛选重要特征(和标签y求pearson相关系数感觉不太行)
top_k_pearson = 6
y_trans = y_train.reshape(-1)
pearson_matrix = np.corrcoef(X_train.T, y_trans)
correlations = np.abs(pearson_matrix[0, 1:])
top_k_indices_pearson = np.argsort(correlations)[::-1][:top_k_pearson]
print("与标签y的Pearson相关系数绝对值最大的前{}个特征的索引：{}".format(top_k_pearson, top_k_indices_pearson))

# SUBSECTION 借助互信息筛选重要特征(单个特征和标签之间的互信息)
top_k_mi = 6
y_trans_mi = y_train.reshape(-1)
mi = mutual_info_regression(X_train, y_trans_mi)
top_k_indices = np.argsort(mi)[::-1][:top_k_mi]
print("互信息最多的前{}个特征的索引：{}".format(top_k_mi, top_k_indices))

# SUBSECTION 借助lasso筛选重要特征(特征的联合分布和标签之间的线性相关性)
alpha = 0.0001
top_k_lasso = 6
lasso = Lasso(alpha, max_iter=10000, tol=0.01)
lasso.fit(X_train, y_train)
coef = lasso.coef_
coef_abs = abs(coef)
top_k_indices = np.argsort(coef_abs)[::-1][:top_k_lasso]
print("lasso绝对值最大的前{}个属性的索引：{}".format(top_k_lasso, top_k_indices))

# SUBSECTION sklearn库的SelectKBest选择器，借助Fisher检验筛选最有影响力的k个特征
top_k_fisher = 6
selector = SelectKBest(score_func=f_classif, k=top_k_fisher)
y_trans_fisher = y_train.reshape(-1)
X_new = selector.fit_transform(X_train, y_trans_fisher)
# 获取被选中的特征的索引
selected_feature_indices = selector.get_support(indices=True)
print("SelectKBest选择器借助Fisher检验的最有影响力的{}个特征的索引：{}".format(top_k_fisher, selected_feature_indices))

# SUBSECTION 借助CART决策树筛选最有影响力的k个特征
top_k_cart = 6
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
# 获取特征重要性得分
feature_importance = clf.feature_importances_
# 根据重要性得分降序排序
sorted_indices = np.argsort(feature_importance)[::-1]
# 根据重要性得分降序排序
top_k_features = sorted_indices[:top_k_cart]
print("CART决策树检验的最有影响力的的前{}个属性的索引：{}".format(top_k_cart, top_k_features))

# SUBSECTION sklearn库SelectFromModel选择器,它可以与任何具有coef_ 或 feature_importances_ 属性（如随机森林和决策树模型）的评估器一起使用来选择特征
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# 使用SelectFromModel来选择重要特征
sfm = SelectFromModel(clf, threshold='mean', prefit=True)
X_selected = sfm.transform(X_train)
# 获取选择的特征索引
selected_idx = sfm.get_support(indices=True)
# 打印选择的特征索引
print("SelectFromModel选择器选择的特征索引:", selected_idx)

# SUBSECTION 借助wrapper(包装)方法生成特征子集
model = LinearRegression()
# 初始化 RFE 特征选择器，选择要保留的特征数量
rfe = RFE(model, n_features_to_select=6)
# 拟合 RFE 特征选择器
rfe.fit(X_train, y_train)
# 输出选择的特征
indices = np.where(rfe.support_)[0]
print("wrapper(包装)方法选择的特征:", indices)
# 输出特征排名
print("wrapper(包装)方法下的特征排名:", rfe.ranking_)

# SUBSECTION 基于XGBoost模型以及XGB的特征重要性
top_k_xgboost = 6
feature_names = drybean.columns.values.tolist()
gbtree = XGBClassifier(n_estimators=2000, max_depth=4, learning_rate=0.05, n_jobs=8)
gbtree.set_params(eval_metric='auc', early_stopping_rounds=100)
X_train_df = pd.DataFrame(X_train, columns=feature_names[:16])
X_test_df = pd.DataFrame(X_test, columns=feature_names[:16])
gbtree.fit(X_train, y_train,eval_set=[(X_test, y_test)], verbose=100)
feature_importances = gbtree.feature_importances_
top_k_indices = np.argpartition(-feature_importances, top_k_xgboost)[:top_k_xgboost]
print("XGBoost检验的最有影响力的的前{}个属性的索引：{}".format(top_k_xgboost, top_k_indices))
# SUBSECTION 无模型方法中的Permutation Feature Importance-slearn
result = permutation_importance(gbtree, X_train, y_train, n_repeats=10,random_state=42)
feature_importance = result.importances_mean
top_k_permutation = np.argpartition(-feature_importances, top_k_xgboost)[:top_k_xgboost]
print("Permutation Feature Importance-slearn检验的最有影响力的的前{}个属性的索引：{}".format(top_k_xgboost, top_k_indices))

# SUBSECTION 基于Partial Dependency Plots预测和标签相关的特征重要性(预测单个特征和标签的关联，暂时无用)
# SUBSECTION 深入到单个样本，分析特征变化对单个样本的影响。求解过程和上述PDP类似，但是ICE会给出每一个样本的预测值(暂时无用，可能会有用)

# SUBSECTION LIME(Local Interpretable Model-Agnostic Explanation), 通过扰动输入样本（perturb the input），来对模型的预测结果进行解释。
# 所选的进行LIME分析的样本
i = 16
top_k_LIME = 6
np.random.seed(1)
explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3)
predict_fn = lambda x: gbtree.predict_proba(x)
exp = explainer.explain_instance(X_train[i], predict_fn, num_features=6)
# 获取最具影响力的特征及其权重
# select_feature_list = []
top_features = exp.as_list()
# print("最具影响力的特征及其权重:")
# for feature, weight in top_features:
#     select_feature_list.append(feature)
#     print(f"{feature}: {weight}")
important_features = []
for feature_set in top_features:
    feature_long = feature_set[0]
    for feature in feature_names:
        if set(feature).issubset(set(feature_long)):
            important_features.append(feature)
            break
important_feature_indices = [feature_names.index(feature_name) for feature_name in important_features]
# print("LIME检验的最有影响力的的前{}个属性的索引：{}".format(top_k_LIME, select_feature_list))
print("LIME检验的最有影响力的的前{}个属性的索引：{}".format(top_k_LIME, important_feature_indices))

# SUBSECTION 借助SHAP(Shapley Additive explanation)值得到有影响力的特征(报错，XGBoost和shap训练数据维度不适配，原因参见https://github.com/shap/shap/issues/580)
# 所选的进行SHAP分析的样本
# i = 16
# top_k_shap = 6
# explainer = shap.TreeExplainer(gbtree)
# shap_values = explainer.shap_values(X_test, y=y_test)
# # 对一个样本求shap值，各个特征对output所带来的变化
# shap.force_plot(explainer.expected_value, shap_values[i,:], X_test[16], matplotlib=True)
# # 计算绝对平均 SHAP 值
# mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
# # 将特征按照绝对平均 SHAP 值的大小进行排序
# sorted_indices = np.argsort(mean_abs_shap_values)[::-1]  # 逆序排序
# # 获取影响最大的特征名称
# top_features = X_test.columns[sorted_indices]
# # 输出影响最大的特征
# print("借助平均shap值判别的影响力较大的特征：", top_features)
# print("shap检验的最有影响力的的前{}个属性的索引：{}".format(top_k_shap, sorted_indices[top_k_shap]))

# SECTION SVM模型训练和分类准确度
svm_model = svm.SVC()
# svm_model = svm.SVC(C=10)  # 默认使用 RBF 核函数（径向基函数），即高斯核函数
# svm_model = svm.SVC(C=2, gamma=0.1)
# svm_model = svm.SVC(kernel='linear')  # 线性核函数
# svm_model = svm.SVC(kernel='poly', degree=3, coef0=1)  # 多项式核函数
# svm_model = svm.SVC(kernel='sigmoid', gamma=0.1, coef0=0.5)  # sigmoid核函数

svm_model.fit(X_train, y_train)
train_label_pred = svm_model.predict(X_train)
wrong_classified_indices = np.where(y_train != svm_model.predict(X_train))[0]
X_train_outliers = X_train[train_outliers_index]
y_train_outliers_numpy = y_train[train_outliers_index]
y_train_outliers = pd.Series(y_train_outliers_numpy)
if not y_train_outliers.empty:
    print("训练集中异常值的SVM分类准确度：" + str(accuracy_score(y_train_outliers, svm_model.predict(X_train_outliers))))

# SECTION 使用sklearn库中的hinge损失函数
decision_values = svm_model.decision_function(X_train)
predicted_labels = np.argmax(decision_values, axis=1)
print("基于hinge损失的训练集上标签的SVM分类准确度：" + str(accuracy_score(y_train, predicted_labels)))
# 计算训练样本的平均损失
train_losses = hinge_loss(y_train, decision_values, labels=np.unique(y_train))
print("整个训练集下的平均hinge损失：", train_losses)
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
# 判定异常：假设阈值为 1，超过此值即认为是异常
anomalies = np.where(hinge_loss > 1)[0]
soft_anomalies = np.where((hinge_loss > 0) & (hinge_loss <= 1))[0]
correct_class = np.where(hinge_loss == 0)[0]
# 输出训练集中outliers中具有较高hinge损失的样本索引
# 训练数据中的异常值，导致SVM分类错误的样本
inter_anomalies = list(set(train_outliers_index) & set(anomalies))
# 测试数据中的潜在异常值，未导致SVM分类错误，但正确分类的预测值与剩余错误分类的最大预测值相差不足阈值1
inter_soft_anomalies = list(set(train_outliers_index) & set(soft_anomalies))
# 测试数据中的潜在异常值，未导致SVM分类错误，且正确分类的预测值与剩余错误分类的最大预测值相差超过阈值1
inter_correct_class = list(set(train_outliers_index) & set(correct_class))

print("*" * 100)
print("训练集中SVM具有较高hinge损失函数的样本数量：", len(anomalies))
# print("训练集中SVM的hinge损失函数高于1的样本索引：", anomalies)
print("训练集中SVM分类错误的样本数量：", len(wrong_classified_indices))
# print("训练集中SVM分类错误的样本索引：", wrong_classified_indices)
intersection = np.intersect1d(anomalies, wrong_classified_indices)
diff_elements = np.setdiff1d(wrong_classified_indices, intersection)
print("分类错误的样本中未被hinge阈值大于1识别的样本索引：", diff_elements)
print("hinge损失大于1的样本和分类错误的样本的交集数量：", len(intersection))
print("该交集所占分类错误的样本的比例：", len(intersection)/len(wrong_classified_indices))
print("*" * 100)
print("训练集的异常值中损失函数高于阈值1的样本索引：", inter_anomalies)
print("训练集的异常值中损失函数在0和阈值1之间的样本索引：", inter_soft_anomalies)
print("训练集的异常值中损失函数为0的样本索引：", inter_correct_class)

# 对训练数据进行处理后重新训练SVM模型

# SECTION 原始数据中的svm分类准确度
print("*" * 100)
print("原始训练集中SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("原始测试集中SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉SVM训练数据中hinge损失函数高于1，且被判定为异常值的样本，重新在处理后的训练数据上训练SVM
# 生成布尔索引，为要删除的行创建布尔值数组
mask = np.ones(len(X_train), dtype=bool)
mask[inter_anomalies] = False
# 使用布尔索引删除那些既被判定为异常值，又使得hinge损失高于1的样本
X_train_split = X_train[mask]
y_train_split = y_train[mask]
# 重新训练SVM模型
svm_model_split = svm.SVC()
svm_model_split.fit(X_train_split, y_train_split)
print("*" * 100)
print("去除同时满足异常和损失高于阈值的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_split, svm_model_split.predict(X_train_split))))
print("去除同时满足异常和损失高于阈值的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_split.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉SVM训练数据中hinge损失函数高于1的样本，重新在处理后的训练数据上训练SVM
mask_h = np.ones(len(X_train), dtype=bool)
mask_h[anomalies] = False
X_train_h = X_train[mask_h]
y_train_h = y_train[mask_h]
svm_model_h = svm.SVC()
svm_model_h.fit(X_train_h, y_train_h)
print("*" * 100)
print("去除损失高于阈值的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_h, svm_model_h.predict(X_train_h))))
print("去除损失高于阈值的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_h.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉SVM训练数据中被判定为异常值的样本，重新在处理后的训练数据上训练SVM
mask_o = np.ones(len(X_train), dtype=bool)
mask_o[train_outliers_index] = False
X_train_o = X_train[mask_o]
y_train_o = y_train[mask_o]
svm_model_o = svm.SVC()
svm_model_o.fit(X_train_o, y_train_o)
print("*" * 100)
print("去除判定为异常的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_o, svm_model_o.predict(X_train_o))))
print("去除判定为异常的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_o.predict(X_test))))
print("*" * 100)