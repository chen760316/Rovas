"""
ML损失函数汇总
"""

import numpy as np
import matplotlib.pyplot as plt

"""
感知器损失 （0-1损失）
"""
# Generate random binary classification dataset
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 samples with 2 features each
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Labels: 1 if sum of features > 1, else 0

# Assume our model always predicts 1 for demonstration
y_pred = np.ones_like(y)

# Calculate perceptron loss
loss = np.mean(np.abs(y - y_pred))

# Print the loss value
print(f"Perceptron Loss: {loss}")

# Plot data points and decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Binary Classification Dataset and Decision Boundary")

# Plot the decision boundary (a simple straight line here)
x_line = np.linspace(0, 1.2, 100)
y_line = 1 - x_line
plt.plot(x_line, y_line, 'k--', label="Decision Boundary")
plt.legend()
plt.show()

"""
均方误差 MSE
"""
# 真实值和预测值
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.2, 2.5, 3.7, 4.1, 5.3])

# 计算均方误差
mse = np.mean((y_true - y_pred) ** 2)
print("MSE:", mse)

# 绘制真实值和预测值的散点图
plt.scatter(y_true, y_pred)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2)  # 绘制直线y=x
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Scatter plot of True vs Predicted Values')
plt.show()

"""
平均绝对误差 MAE
"""
# 生成随机数据
np.random.seed(0)
n = 50
X = np.linspace(0, 10, n)
y_true = 2 * X + 1 + np.random.normal(0, 1, n)  # 真实的目标值，包含随机噪音
y_pred = 2 * X + 1.5  # 模拟的预测值

# 计算MAE
mae = np.mean(np.abs(y_true - y_pred))

# 绘制数据点和预测线
plt.scatter(X, y_true, label='Actual', color='b')
plt.plot(X, y_pred, label='Predicted', color='r')
plt.title(f'MAE = {mae:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

"""Huber损失"""
def huber_loss(y_true, y_pred, delta):
    error = np.abs(y_true - y_pred)
    if error <= delta:
        return 0.5 * error ** 2
    else:
        return delta * error - 0.5 * delta ** 2

# 生成一些样本数据
np.random.seed(0)
X = np.linspace(0, 10, 100)
y_true = 2 * X + 1 + np.random.normal(0, 1, 100)

# 定义模型预测值
y_pred = 2 * X + 1

# 计算不同delta值下的损失
deltas = [1.0, 2.0, 5.0]
losses = []

for delta in deltas:
    loss = [huber_loss(y_true[i], y_pred[i], delta) for i in range(len(X))]
    losses.append(loss)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
for i, delta in enumerate(deltas):
    plt.plot(X, losses[i], label=f'Huber Loss (delta={delta})')

plt.xlabel('X')
plt.ylabel('Loss')
plt.title('Huber Loss for Different Delta Values')
plt.legend()
plt.grid(True)
plt.show()

"""
交叉熵损失函数 CEL
"""
def cross_entropy_loss(y_true, p_pred):
    epsilon = 1e-10  # 添加一个小的常数以避免log(0)计算错误
    return -np.sum(y_true * np.log(p_pred + epsilon), axis=1)

num_samples = 1000
num_classes = 5
np.random.seed(42)
y_true = np.eye(num_classes)[np.random.choice(num_classes, num_samples)]  # 生成随机的one-hot标签,真实类别标签为1，其余标签为0
p_pred = np.random.rand(num_samples, num_classes)  # 模型预测的概率

loss = cross_entropy_loss(y_true, p_pred)
# 计算平均损失
average_loss = np.mean(loss)
# 绘制损失函数图形
plt.plot(range(num_samples), loss, 'bo', markersize=2)
plt.xlabel('Sample')
plt.ylabel('Cross-Entropy Loss')
plt.title('Cross-Entropy Loss for each Sample')
plt.axhline(average_loss, color='r', linestyle='--', label='Average Loss')
plt.legend()
plt.show()
print(f'Average Loss: {average_loss}')

"""
对数损失 LL
"""
def log_loss(y_true, p_pred):
    epsilon = 1e-10  # 添加一个小的常数以避免log(0)计算错误
    return - (y_true * np.log(p_pred + epsilon) + (1 - y_true) * np.log(1 - p_pred + epsilon))


# 模拟数据
num_samples = 1000
np.random.seed(42)
y_true = np.random.randint(2, size=num_samples)  # 随机生成0和1的实际标签
p_pred = np.random.rand(num_samples)  # 模型预测的概率

loss = log_loss(y_true, p_pred)

# 计算平均损失
average_loss = np.mean(loss)

# 绘制损失函数图形
plt.plot(range(num_samples), loss, 'bo', markersize=2)
plt.xlabel('Sample')
plt.ylabel('Log Loss')
plt.title('Log Loss for each Sample')
plt.axhline(average_loss, color='r', linestyle='--', label='Average Loss')
plt.legend()
plt.show()

print(f'Average Loss: {average_loss}')

"""
多类别交叉熵损失（同交叉熵损失）
"""

"""
二分类交叉熵损失（同对数损失）
"""

"""
余弦相似度损失函数(CosineEmbeddingLoss)
"""
import torch
import torch.nn as nn

torch.manual_seed(20)
cosine_loss = nn.CosineEmbeddingLoss(margin=0.2)
a = torch.randn(100, 128, requires_grad=True)
b = torch.randn(100, 128, requires_grad=True)
print(a.size())
print(b.size())
y = 2 * torch.empty(100).random_(2) - 1
output = cosine_loss(a, b, y)
print(output.item())
triplet_loss = nn.CosineEmbeddingLoss(margin=0.2, reduction="none")
output = triplet_loss(a, b, y)
print(output)

"""
HingeEmbeddingLoss
"""
import torch
import torch.nn as nn

torch.manual_seed(20)
hinge_loss = nn.HingeEmbeddingLoss(margin=0.2)
a = torch.randn(100, 128, requires_grad=True)
b = torch.randn(100, 128, requires_grad=True)
x = 1 - torch.cosine_similarity(a, b)
# 定义a与b之间的距离为x
print(x.size())
y = 2 * torch.empty(100).random_(2) - 1
output = hinge_loss(x, y)
print(output.item())

hinge_loss = nn.HingeEmbeddingLoss(margin=0.2, reduction="none")
output = hinge_loss(x, y)
print(output)