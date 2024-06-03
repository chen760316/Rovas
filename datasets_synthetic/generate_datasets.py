"""
xuhongzuo库中合成数据生成方式及示例
"""

# unsupervised methods
from deepod.models import RCA
from deepod.models.tabular import GOAD
from deepod.utils.data import generate_data
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""
GOAD无监督模型训练和预测
"""

n_train = 500
n_test = 100
n_features = 10
contamination = 0.1
random_state = 42
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64

X_train, X_test, y_train, y_test = generate_data(
    n_train=n_train, n_test=n_test, n_features=n_features,
    contamination=contamination, random_state=random_state, offset=10)
train = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
test = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
synthetic_dataset = np.concatenate((train, test), axis=0)
result = np.round(synthetic_dataset * 10)
df = pd.DataFrame(result, columns=['feature' + str(i) for i in range(result.shape[1])])
df.rename(columns={df.columns[-1]: 'label'}, inplace=True)
df.to_csv('synthetic_dataset.csv', index=False)





