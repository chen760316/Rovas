"""
按照靳老师的想法利用高斯混合模型(GMM)判别单列中的异常
自定义高斯混合模型
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

np.random.seed(None)

class MyGMM(object):
    def __init__(self, K=3):
        """
        高斯混合模型，用EM算法进行求解
        :param K: 超参数，分类类别

        涉及到的其它参数：
        :param N: 样本量
        :param D: 单个样本的维度
        :param alpha: 模型参数，高斯函数的系数，决定高斯函数的高度，维度（K）
        :param mu: 模型参数，高斯函数的均值，决定高斯函数的中型位置，维度（K,D）
        :param Sigma: 模型参数，高斯函数的方差矩阵，决定高斯函数的形状，维度（K,D,D）
        :param gamma: 模型隐变量，决定单个样本具体属于哪一个高斯分布，维度(N,K)
        """
        self.K = K
        self.params = {
            'alpha': None,
            'mu': None,
            'Sigma': None,
            'gamma': None
        }

        self.N = None
        self.D = None

    def __init_params(self):
        # alpha 需要满足和为1的约束条件
        alpha = np.random.rand(self.K)
        alpha = alpha / np.sum(alpha)
        mu = np.random.rand(self.K, self.D)
        Sigma = np.array([np.identity(self.D) for _ in range(self.K)])
        # 虽然gamma有约束条件，但是第一步E步时会对此重新赋值，所以可以随意初始化
        gamma = np.random.rand(self.N, self.K)

        self.params = {
            'alpha': alpha,
            'mu': mu,
            'Sigma': Sigma,
            'gamma': gamma
        }

    def _gaussian_function(self, y_j, mu_k, Sigma_k):
        '''
        计算高纬度高斯函数
        :param y_j: 第j个观测值
        :param mu_k: 第k个mu值
        :param Sigma_k: 第k个Sigma值
        :return:
        '''
        # 先取对数
        n_1 = self.D * np.log(2 * np.pi)
        # 计算数组行列式的符号和（自然）对数。
        _, n_2 = np.linalg.slogdet(Sigma_k)

        # 计算矩阵的（乘法）逆矩阵。
        n_3 = np.dot(np.dot((y_j - mu_k).T, np.linalg.inv(Sigma_k)), y_j - mu_k)

        # 返回是重新取指数抵消前面的取对数操作
        return np.exp(-0.5 * (n_1 + n_2 + n_3))

    def _E_step(self, y):
        alpha = self.params['alpha']
        mu = self.params['mu']
        Sigma = self.params['Sigma']

        for j in range(self.N):
            y_j = y[j]
            gamma_list = []
            for k in range(self.K):
                alpha_k = alpha[k]
                mu_k = mu[k]
                Sigma_k = Sigma[k]
                gamma_list.append(alpha_k * self._gaussian_function(y_j, mu_k, Sigma_k))

            # 对隐变量进行迭代跟新
            self.params['gamma'][j, :] = np.array([v / np.sum(gamma_list) for v in gamma_list])

    def _M_step(self, y):
        mu = self.params['mu']
        gamma = self.params['gamma']
        for k in range(self.K):
            mu_k = mu[k]
            gamma_k = gamma[:, k]
            gamma_k_j_list = []
            mu_k_part_list = []
            Sigma_k_part_list = []
            for j in range(self.N):
                y_j = y[j]
                gamma_k_j = gamma_k[j]
                gamma_k_j_list.append(gamma_k_j)

                # mu_k的分母的分母列表
                mu_k_part_list.append(gamma_k_j * y_j)

                # Sigma_k的分母列表
                Sigma_k_part_list.append(gamma_k_j * np.outer(y_j - mu_k, (y_j - mu_k).T))

            # 对模型参数进行迭代更新
            self.params['mu'][k] = np.sum(mu_k_part_list, axis=0) / np.sum(gamma_k_j_list)
            self.params['Sigma'][k] = np.sum(Sigma_k_part_list, axis=0) / np.sum(gamma_k_j_list)
            self.params['alpha'][k] = np.sum(gamma_k_j_list) / self.N

    def fit(self, y, max_iter=100):
        y = np.array(y)
        self.N, self.D = y.shape
        self.__init_params()

        for _ in range(max_iter):
            self._E_step(y)
            self._M_step(y)

def get_samples(n_ex=1000, n_classes=3, n_in=2, seed=None):
    # 生成100个样本，为了能够在二维平面上画出图线表示出来，每个样本的特征维度设置为2
    y, _ = make_blobs(
        n_samples=n_ex, centers=n_classes, n_features=n_in, random_state=seed)
    return y

def run_my_model():
    gmm_model = MyGMM()
    y = get_samples()
    gmm_model.fit(y)

    max_index = np.argmax(gmm_model.params['gamma'], axis=1)
    print(max_index)

    k1_list = []
    k2_list = []
    k3_list = []

    for y_i, index in zip(y, max_index):
        if index == 0:
            k1_list.append(y_i)
        elif index == 1:
            k2_list.append(y_i)
        else:
            k3_list.append(y_i)
    k1_list = np.array(k1_list)
    k2_list = np.array(k2_list)
    k3_list = np.array(k3_list)

    plt.scatter(k1_list[:, 0], k1_list[:, 1], c='red')
    plt.scatter(k2_list[:, 0], k2_list[:, 1], c='blue')
    plt.scatter(k3_list[:, 0], k3_list[:, 1], c='green')
    plt.show()

if __name__ == "__main__":
    run_my_model()