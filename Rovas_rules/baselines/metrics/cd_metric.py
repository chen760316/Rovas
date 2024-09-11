"""
Wilcoxon-Holm 方法 是一种用于多重比较的统计方法，常用于对多个分类器或算法的性能进行统计显著性测试。
它是 Wilcoxon Rank-Sum Test 和 Holm-Bonferroni 校正 方法的组合，
用于控制假阳性率并进行多重比较校正。
临界差异图（Critical Difference Plot, CD 图） 是一种可视化工具，用于展示多种算法在性能上的显著性差异。
Wilcoxon Rank-Sum Test
    是一种非参数检验方法，用于比较两组样本的中位数是否存在显著差异。
    不要求数据符合正态分布，适用于小样本和非正态分布的数据。
Holm-Bonferroni 校正
    用于调整多重比较中产生的假阳性率。
    Holm 方法逐步调整每个假设的显著性水平，从最小的 p 值开始，对每个比较进行校正。

"""

# 性能数据：每个分类器的性能结果，例如准确率、F1 分数等。
# 统计测试：使用 Wilcoxon Rank-Sum Test 来计算每对分类器的 p 值，并通过 Holm-Bonferroni 方法进行校正。
# CD 图：将每个分类器的平均性能绘制在图上，并用不同的颜色或符号标出显著差异。

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums

# 假设有三个分类器的性能数据
data = {
    'Classifier A': np.array([0.85, 0.88, 0.90, 0.86, 0.87]),
    'Classifier B': np.array([0.80, 0.82, 0.84, 0.81, 0.83]),
    'Classifier C': np.array([0.75, 0.78, 0.77, 0.76, 0.74])
}

# 计算每对分类器的 Wilcoxon Rank-Sum Test p 值
p_values = {}
classifiers = list(data.keys())
for i in range(len(classifiers)):
    for j in range(i + 1, len(classifiers)):
        c1 = classifiers[i]
        c2 = classifiers[j]
        _, p = ranksums(data[c1], data[c2])
        p_values[(c1, c2)] = p

# 打印 p 值
for pair, p in p_values.items():
    print(f'p-value for {pair}: {p}')

# 绘制 CD 图（示例）
# 这里简化为直接绘制算法性能的条形图
fig, ax = plt.subplots()
bars = []
for i, (clf, scores) in enumerate(data.items()):
    bar = ax.bar(i, np.mean(scores), yerr=np.std(scores), capsize=5, label=clf)
    bars.append(bar)

ax.set_xticks(range(len(data)))
ax.set_xticklabels(data.keys())
ax.set_ylabel('Mean Performance')
ax.set_title('Critical Difference Plot (CD Plot)')
ax.legend()

plt.show()
