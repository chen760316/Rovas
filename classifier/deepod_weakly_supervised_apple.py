# unsupervised methods
from deepod.models.tabular import DevNet
from deepod.utils.data import generate_data
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

"""
kaggle datasets
"""
file_path = "..\\kaggle datasets\\Apple Quality\\apple_quality.csv"
apple_quality = pd.read_csv(file_path)
apple_quality['Quality'] = apple_quality['Quality'].replace({'good': 1, 'bad': 0})
apple_indicates = apple_quality.drop(["Quality", "A_id"], axis=1)
X_train, X_test, y_train, y_test = \
    train_test_split(apple_indicates, apple_quality["Quality"],
                     test_size=0.3, random_state=1)
n_train = X_train.shape[0]
n_test = X_test.shape[0]
n_features = X_train.shape[1]
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
hidden_dims = 20
random_state = 42
clf = DevNet(epochs=epochs, hidden_dims=hidden_dims, device=device,
                  random_state=random_state)
clf.fit(X_train.values, y_train.values)
scores = clf.decision_function(X_test.values)
pred_labels, confidence = clf.predict(X_test.values, return_confidence=True)
print("训练集中异常值判定阈值为：", clf.threshold_)
print("测试样本得分，样本标签值，预测置信度如下")
outliers_index = []
for i in range(len(scores)):
    print(scores[i], pred_labels[i], confidence[i])
    if pred_labels[i] == 1:
        outliers_index.append(i+1)
print("测试集中异常值索引为：", outliers_index)
print("测试集中异常值所占比例为：", len(outliers_index)/len(scores))
svm_model = svm.SVC(C=10)
# svm_model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)
print("SVM训练模型评分：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("SVM测试模型评分：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
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
