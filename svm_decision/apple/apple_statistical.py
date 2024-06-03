"""
在Apple数据集上测试统计方法
"""
# unsupervised methods
from deepod.models import REPEN, SLAD, ICL, RDP, NeuTraL, DeepSAD, FeaWAD, RoSAS, PReNet
from deepod.models.tabular import DevNet
from deepod.models.tabular import GOAD
from deepod.models.tabular import RCA
from deepod.models.tabular import DeepSVDD
from deepod.utils.data import generate_data
from scipy import stats
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
import numpy as np

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42
hidden_dims=20
"""
kaggle datasets
"""
# file_path = "..\\kaggle datasets\\Apple Quality\\apple_quality.csv"
file_path = "../../kaggle_datasets/Apple_Quality/apple_quality.csv"
apple_quality = pd.read_csv(file_path)
apple_quality['Quality'] = apple_quality['Quality'].replace({'good': 1, 'bad': 0})
apple_indicates = apple_quality.drop(["Quality", "A_id"], axis=1)
X_train, X_test, y_train, y_test = \
    train_test_split(apple_indicates, apple_quality["Quality"],
                     test_size=0.5, random_state=1)
X_train = X_train.values
X_test = X_test.values
"""
四分位极差法
"""
# q1 = apple_quality['Size'].quantile(0.25)
# q3 = apple_quality['Size'].quantile(0.75)
# iqr = q3 - q1
#
# lower_bound = q1 - 1.5 * iqr
# upper_bound = q3 + 1.5 * iqr
#
# outliers_quartile = apple_quality[(apple_quality['Size'] < lower_bound)
#                                   | (apple_quality['Size'] > upper_bound)]
# outliers_index = outliers_quartile.index
"""
标准偏差法
"""
# mean = apple_quality.Size.mean()
# std = apple_quality.Size.std()
# upper_bound = mean + 3 * std
# lower_bound = mean - 3 * std
# outliers_standard = apple_quality[(apple_quality['Size'] < lower_bound) | (apple_quality['Size'] > upper_bound)]
# outliers_index = outliers_standard.index
"""
Z-分数法
"""
apple_quality['Size_zscore'] = stats.zscore(apple_quality['Size'])
outliers_Z_score = apple_quality[(apple_quality['Size_zscore'] > 3) | (apple_quality['Size_zscore'] < -3)]
print(outliers_Z_score)
outliers_index = outliers_Z_score.index

print("数据集中异常值索引为：", outliers_index)
print(len(apple_quality))
print("数据集中异常值所占比例为：", len(outliers_index)/len(apple_quality))
svm_model = svm.SVC(C=10)
# svm_model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)
print("训练集SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
X_outliers = apple_indicates.to_numpy()[outliers_index]
y_outliers = apple_quality['Quality'].to_numpy()[outliers_index]
print("数据集中outliers的SVM分类准确度：" + str(accuracy_score(y_outliers, svm_model.predict(X_outliers))))



pca = PCA(n_components=2)
pca.fit(apple_indicates)
iris_indicates_reduction = pca.transform(apple_indicates)  # 降维后的结果
X_train_reduction, X_test_reduction, y_train_reduction, y_test_reduction = train_test_split(iris_indicates_reduction, apple_quality['Quality'], random_state=1, train_size=0.7, test_size=0.3)  # 降维后的测试集划分
# 训练集和测试集的预测结果
trainPredict = svm_model.predict(X_train).reshape(-1, 1)
testPredict = svm_model.predict(X_test).reshape(-1, 1)
# 将预测结果进行展示,首先画出预测点，再画出分类界面
# 画图例和点集
apple_indicates_reduction_row__min, apple_indicates_reduction_row_max = iris_indicates_reduction[:, 0].min(), iris_indicates_reduction[:, 0].max()  # x轴范围
apple_indicates_reduction_column_min, apple_indicates_reduction_column_max = iris_indicates_reduction[:, 1].min(), iris_indicates_reduction[:, 1].max()  # y轴范围
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])  # 设置点集颜色格式
cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])  # 设置边界颜色
plt.rcParams['font.family'] = 'Arial'
plt.xlabel('fea1', fontsize=13)  # x轴标注
plt.ylabel('fea2', fontsize=13)  # y轴标注
plt.xlim(apple_indicates_reduction_row__min, apple_indicates_reduction_row_max)  # x轴范围
plt.ylim(apple_indicates_reduction_column_min, apple_indicates_reduction_column_max)  # y轴范围
plt.title('SVM result')  # 标题
plt.scatter(iris_indicates_reduction[:200, 0], iris_indicates_reduction[:200, 1], c=apple_quality['Quality'][:200], s=30, cmap=cm_dark)  # 画出测试点
plt.scatter(X_test_reduction[:200, 0], X_test_reduction[:200, 1], c=testPredict[:200, 0], s=80, edgecolors='k', marker='*', zorder=2, cmap=cm_dark)  # 画出测试点，并将预测点圈出(注意这里是测试集的预测标签值)
plt.show()
