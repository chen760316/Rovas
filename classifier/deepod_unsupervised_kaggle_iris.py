from sklearn import svm  # 引入svm包
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA

file_path = '..\\kaggle datasets\\Iris Species\\Iris.csv'
iris = pd.read_csv(file_path)
iris['Species'] = iris['Species'].replace({'Iris-setosa': 2, 'Iris-versicolor': 1, 'Iris-virginica': 0})
iris_indicates = iris.drop(['Species', 'Id'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(iris_indicates, iris['Species'], random_state=1, train_size=0.7, test_size=0.3)
# 创建svm分类器并进行训练：首先，利用sklearn中的SVC（）创建分类器对象，其中常用的参数有C（惩罚力度）、kernel（核函数）、gamma（核函数的参数设置）、decision_function_shape（因变量的形式），再利用fit()用训练数据拟合分类器模型。
'''C越大代表惩罚程度越大，越不能容忍有点集交错的问题，但有可能会过拟合（defaul C=1）；
kernel常规的有‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ ，默认的是rbf；
gamma是核函数为‘rbf’, ‘poly’ 和 ‘sigmoid’时的参数设置，其值越小，分类界面越连续，其值越大，分类界面越“散”，分类效果越好，但有可能会过拟合，默认的是特征个数的倒数；
decision_function_shape='ovr'时，为one v rest（一对多），即一个类别与其他类别进行划分，等于'ovo'时，为one v one（一对一），即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
'''
model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
model.fit(X_train, y_train.ravel())  # ravel函数在降维时默认是行序优先
# 利用classifier.score（）分别计算训练集和测试集的准确率。
train_score = model.score(X_train, y_train)
print("训练集：", train_score)
test_score = model.score(X_test, y_test)
print("测试集：", test_score)
# 决策函数的查看（可省略）
# print('train_decision_function:\n',model.decision_function(train_data))#（90，3）
# pca降维
# 直接使用sklearn中的PCA进行降维

pca = PCA(n_components=2)
pca.fit(iris_indicates)
iris_indicates_reduction = pca.transform(iris_indicates)  # 降维后的结果
X_train_reduction, X_test_reduction, y_train_reduction, y_test_reduction = train_test_split(iris_indicates_reduction, iris['Species'], random_state=1, train_size=0.7, test_size=0.3)  # 降维后的测试集划分

# 训练集和测试集的预测结果
trainPredict = model.predict(X_train).reshape(-1, 1)
testPredict = model.predict(X_test).reshape(-1, 1)
# 将预测结果进行展示,首先画出预测点，再画出分类界面
# 画图例和点集
iris_indicates_reduction_row__min, iris_indicates_reduction_row_max = iris_indicates_reduction[:, 0].min(), iris_indicates_reduction[:, 0].max()  # x轴范围
iris_indicates_reduction_column_min, iris_indicates_reduction_column_max = iris_indicates_reduction[:, 1].min(), iris_indicates_reduction[:, 1].max()  # y轴范围
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])  # 设置点集颜色格式
cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])  # 设置边界颜色
plt.rcParams['font.family'] = 'Arial'
plt.xlabel('fea1', fontsize=13)  # x轴标注
plt.ylabel('fea2', fontsize=13)  # y轴标注
plt.xlim(iris_indicates_reduction_row__min, iris_indicates_reduction_row_max)  # x轴范围
plt.ylim(iris_indicates_reduction_column_min, iris_indicates_reduction_column_max)  # y轴范围
plt.title('SVM result')  # 标题
plt.scatter(iris_indicates_reduction[:, 0], iris_indicates_reduction[:, 1], c=iris['Species'], s=30, cmap=cm_dark)  # 画出测试点
plt.scatter(X_test_reduction[:, 0], X_test_reduction[:, 1], c=testPredict[:, 0], s=80, edgecolors='k', marker='*', zorder=2, cmap=cm_dark)  # 画出测试点，并将预测点圈出(注意这里是测试集的预测标签值)
plt.show()