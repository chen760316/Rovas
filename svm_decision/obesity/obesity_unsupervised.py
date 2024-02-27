# unsupervised methods
from deepod.models import REPEN, SLAD, ICL, RDP, NeuTraL
from deepod.models.tabular import DevNet
from deepod.models.tabular import GOAD
from deepod.models.tabular import RCA
from deepod.models.tabular import DeepSVDD
from deepod.utils.data import generate_data
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42
"""
kaggle datasets
"""
file_path = "../../kaggle_datasets/Obesity_prediction/obesity_data.csv"
obesity = pd.read_csv(file_path)
obesity['ObesityCategory'] = obesity['ObesityCategory'].replace({'Obese': 3, 'Overweight': 2,\
                                                'Normal weight': 1, 'Underweight': 0})
obesity['Gender'] = obesity['Gender'].replace({'Male': 1, 'Female': 0})
obesity_indicates = obesity.drop(["ObesityCategory"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(obesity_indicates, obesity["ObesityCategory"],\
                     test_size=0.5, random_state=1)
X_train = X_train.values
X_test = X_test.values
"""GOAD"""
# clf = GOAD(epochs=epochs, device=device, n_trans=n_trans)
# clf.fit(X_train, y=None)
"""DeepSVDD"""
# clf = DeepSVDD(epochs=epochs, device=device, random_state=random_state)
# clf.fit(X_train, y=None)
"""RCA"""
# clf = RCA(epochs=epochs, device=device, act='LeakyReLU')
# clf.fit(X_train)
"""RePEN"""
# clf = REPEN(epochs=5, device=device)
# clf.fit(X_train)
"""SLAD"""
# clf = SLAD(epochs=2, device=device)
# clf.fit(X_train)
"""ICL"""
clf = ICL(epochs=1, device=device, n_ensemble='auto')
clf.fit(X_train)
"""NeuTraL"""
# clf = NeuTraL(epochs=1, device=device)
# clf.fit(X_train)

scores = clf.decision_function(X_test)
pred_labels, confidence = clf.predict(X_test, return_confidence=True)
print("训练集中异常值判定阈值为：", clf.threshold_)
print("测试样本得分，样本标签值，预测置信度如下")
outliers_index = []
for i in range(len(scores)):
    print(scores[i], pred_labels[i], confidence[i])
    if pred_labels[i] == 1:
        outliers_index.append(i)
print("测试集中异常值索引为：", outliers_index)
print("测试集中异常值所占比例为：", len(outliers_index)/len(scores))
svm_model = svm.SVC(C=10)
# svm_model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)
print("训练集SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
X_test_outliers = X_test[outliers_index]
y_test_outliers_numpy = y_test.to_numpy()[outliers_index]
y_test_outliers = pd.Series(y_test_outliers_numpy)
print("测试集中outliers的SVM分类准确度：" + str(accuracy_score(y_test_outliers, svm_model.predict(X_test_outliers))))
# pca = PCA(n_components=2)
# pca.fit(obesity_indicates)
# obesity_indicates_reduction = pca.transform(obesity_indicates)  # 降维后的结果
# X_train_reduction, X_test_reduction, y_train_reduction, y_test_reduction = train_test_split(obesity_indicates_reduction, obesity['ObesityCategory'], random_state=1, train_size=0.7, test_size=0.3)  # 降维后的测试集划分
# # 训练集和测试集的预测结果
# trainPredict = svm_model.predict(X_train).reshape(-1, 1)
# testPredict = svm_model.predict(X_test).reshape(-1, 1)
# # 将预测结果进行展示,首先画出预测点，再画出分类界面
# # 画图例和点集
# obesity_indicates_reduction_row__min, obesity_indicates_reduction_row_max = obesity_indicates_reduction[:, 0].min(), obesity_indicates_reduction[:, 0].max()  # x轴范围
# obesity_indicates_reduction_column_min, obesity_indicates_reduction_column_max = obesity_indicates_reduction[:, 1].min(), obesity_indicates_reduction[:, 1].max()  # y轴范围
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])  # 设置点集颜色格式
# cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])  # 设置边界颜色
# plt.rcParams['font.family'] = 'Arial'
# plt.xlabel('fea1', fontsize=13)  # x轴标注
# plt.ylabel('fea2', fontsize=13)  # y轴标注
# plt.xlim(obesity_indicates_reduction_row__min, obesity_indicates_reduction_row_max)  # x轴范围
# plt.ylim(obesity_indicates_reduction_column_min, obesity_indicates_reduction_column_max)  # y轴范围
# plt.title('SVM result')  # 标题
# plt.scatter(obesity_indicates_reduction[:200, 0], obesity_indicates_reduction[:200, 1], c=obesity['ObesityCategory'][:200], s=30, cmap=cm_dark)  # 画出测试点
# plt.scatter(X_test_reduction[:200, 0], X_test_reduction[:200, 1], c=testPredict[:200, 0], s=80, edgecolors='k', marker='*', zorder=2, cmap=cm_dark)  # 画出测试点，并将预测点圈出(注意这里是测试集的预测标签值)
# plt.show()
