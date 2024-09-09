"""
邻居选择：KNN修复的质量依赖于邻居的选择。选择合适的K值和距离度量对于修复效果至关重要。K 值通常为奇数，避免对称性问题。
数据特征：KNN修复通常对数据的尺度比较敏感，因此可能需要进行特征标准化。
计算复杂度：KNN在计算上可能较为复杂，尤其是在数据量大的时候，计算最近邻可能比较耗时。
"""
"""
KNN修复流程如下：
一、选择异常值：定位数据集中被标记为异常的数据点。
二、计算邻居：对每个异常值，找到其 K 个最近邻的数据点。通常，你可以使用欧几里得距离或其他适当的距离度量来计算邻居。
三、计算预测值：
3.1 对于回归任务：计算K个邻居的均值或加权均值来替换异常值。
3.2 对于分类任务：计算K个邻居中出现频率最高的类别来替换异常值。
3.3 对于数值型数据，可以使用K个邻居的值进行插值。例如，线性插值或其他插值方法
3.4 对于类别型数据，可以使用K个邻居的类别进行加权投票，决定最可能的类别
四、实施修复：根据KNN模型预测的结果，替换或修复异常值。确保修复后的数据在业务上是合理的，并且对数据的分布没有引入不必要的偏差。
五、验证和评估：在修复异常值后，验证修复的效果
5.1 检查数据分布：确保修复后的数据没有引入新的问题
5.2 模型验证：如果修复是为了改进某个模型的性能，可以用修复后的数据重新训练模型，并比较模型的性能指标（如准确率、均方误差等）来评估修复效果。
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# 示例数据
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # 特征
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100])  # 标签，其中100为异常值

# 识别异常值
is_outlier = np.abs(y - np.mean(y)) > 2 * np.std(y)
X_train = X[~is_outlier]
y_train = y[~is_outlier]

# KNN模型
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

# 预测异常值
X_outliers = X[is_outlier]
y_pred = knn.predict(X_outliers)

# 替换异常值
y[is_outlier] = y_pred

print("修复后的标签:", y)