"""
自编码器（Autoencoder）是一种无监督学习算法，用于数据降维、特征学习以及去噪。
它通过构建一个神经网络模型，学习数据的紧凑表示，从而在压缩和重构数据时保留尽可能多的信息。
自编码器的主要目标是尽量准确地重构输入数据。
一、自编码器的工作原理如下：

1、编码器（Encoder）：
    编码器将输入数据映射到一个低维空间（潜在空间或隐藏层）。这个部分的网络通常由一个或多个全连接层（Dense Layers）组成。
    编码器的输出是数据的紧凑表示，称为编码（encoding）或潜在表示（latent representation）。

2、解码器（Decoder）：
    解码器将编码器生成的低维表示映射回原始数据的空间。这个部分的网络也由一个或多个全连接层组成。
    解码器的输出是对输入数据的重构，目标是使这个重构尽可能接近原始输入数据。

3、重构损失（Reconstruction Loss）：
    自编码器的训练目标是最小化重构损失，即输入数据与重构数据之间的差异。常用的损失函数包括均方误差（MSE）和二元交叉熵（Binary Cross-Entropy）。

二、自编码器的变体

    去噪自编码器（Denoising Autoencoder）：
        通过将输入数据添加噪声进行训练，使得自编码器能够在有噪声的数据上进行有效的重构，从而提高对噪声的鲁棒性。

    稀疏自编码器（Sparse Autoencoder）：
        在编码器的中间层上施加稀疏性约束，使得只有少数的神经元被激活，强调特征学习。

    变分自编码器（Variational Autoencoder, VAE）：
        在潜在空间上施加概率分布假设，通过生成模型生成新数据。VAE 适用于生成任务和数据建模。

    生成对抗自编码器（Adversarial Autoencoder, AAE）：
        将自编码器与生成对抗网络（GAN）结合，使得编码器生成的潜在空间分布更符合预设的概率分布。

三、自编码器的工作流程：
首先，我们在不含异常值的干净数据上训练自编码器，然后对异常值使用自编码器进行重构，以替换这些异常值。
在实际应用中，你可以使用自编码器的重构误差作为异常分数，
如果某个数据点的重构误差超过了预设的阈值，则可以认为该数据点是异常的。
对于这些异常值，你可以选择删除它们，或者使用自编码器的输出（即重构的数据）来替换原始的异常值。

四、自编码器的优缺点：
优点：泛化性强，无监督不需要数据标注
缺点：针对异常识别场景，训练数据需要为正常数据。

"""

import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf
from keras.optimizers import Adam

# 示例数据
data = pd.DataFrame({
    'feature1': [1, 2, 3, 1000, 5],  # 1000 是异常值
    'feature2': [5, 6, 7, 8, 9]
})

# section 自编码器的原始版本（需要修改在干净数据上训练自编码器）

# # 数据标准化
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data)
#
# # 训练自编码器
# input_dim = data_scaled.shape[1]
# input_layer = Input(shape=(input_dim,))
# encoded = Dense(2, activation='relu')(input_layer)
# decoded = Dense(input_dim, activation='sigmoid')(encoded)
#
# autoencoder = Model(input_layer, decoded)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=2, shuffle=True)
#
# # 修复异常值
# reconstructed_data = autoencoder.predict(data_scaled)
#
# # 逆标准化
# data_repaired = scaler.inverse_transform(reconstructed_data)
# data_repaired = pd.DataFrame(data_repaired, columns=data.columns)
#
# print(data_repaired)

# section 自编码器的改进版本（效果有提升）（需要修改在干净数据上训练自编码器）

# data['feature1'] = data['feature1'].replace(1000, data['feature1'].median())
# # 使用鲁棒标准化
# scaler = RobustScaler()
# data_scaled = scaler.fit_transform(data)
#
# # 训练自编码器
# input_dim = data_scaled.shape[1]
# input_layer = Input(shape=(input_dim,))
# encoded = Dense(5, activation='relu')(input_layer)  # 增加隐藏层
# decoded = Dense(input_dim, activation='linear')(encoded)  # 使用线性激活函数
#
# autoencoder = Model(input_layer, decoded)
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')  # 使用均方误差损失
# # autoencoder.compile(optimizer='adam', loss='binary_crossentropy') # 使用二元交叉熵损失
# autoencoder.fit(data_scaled, data_scaled, epochs=100, batch_size=2, shuffle=True)
#
# # 修复异常值
# reconstructed_data = autoencoder.predict(data_scaled)
#
# # 逆标准化
# data_repaired = scaler.inverse_transform(reconstructed_data)
# data_repaired = pd.DataFrame(data_repaired, columns=data.columns)
#
# print(data_repaired)

# section 去噪自编码器，使用干净数据和含噪数据训练去噪自编码器（需要修改在干净数据上训练自编码器）
# # 使用鲁棒标准化
# scaler = RobustScaler()
# data_scaled = scaler.fit_transform(data)
#
# # 添加噪声
# def add_noise(data, noise_factor=0.3):
#     noise = np.random.normal(loc=0, scale=noise_factor, size=data.shape)
#     return np.clip(data + noise, 0., 1.)
#
# data_noisy = add_noise(data_scaled)
#
# # 定义自编码器模型
# input_dim = data_scaled.shape[1]
# input_layer = Input(shape=(input_dim,))
# encoded = Dense(5, activation='relu')(input_layer)  # 编码器层
# decoded = Dense(input_dim, activation='linear')(encoded)  # 使用线性激活函数
#
# autoencoder = Model(input_layer, decoded)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#
# # 训练自编码器
# autoencoder.fit(data_noisy, data_scaled, epochs=100, batch_size=2, shuffle=True)
#
# # 预测（重构）数据
# reconstructed_data = autoencoder.predict(data_noisy)
#
# # 逆标准化
# data_repaired = scaler.inverse_transform(reconstructed_data)
# data_repaired = pd.DataFrame(data_repaired, columns=data.columns)
#
# print("Original Data:")
# print(data)
# print("\nNoisy Data:")
# print(scaler.inverse_transform(data_noisy))
# print("\nRepaired Data:")
# print(data_repaired)
#
# # 可视化（如果数据较少，可以绘图观察）
# plt.figure(figsize=(10, 6))
#
# plt.subplot(1, 3, 1)
# plt.title('Original Data')
# plt.scatter(data['feature1'], data['feature2'])
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
#
# plt.subplot(1, 3, 2)
# plt.title('Noisy Data')
# plt.scatter(scaler.inverse_transform(data_noisy)[:, 0], scaler.inverse_transform(data_noisy)[:, 1])
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
#
# plt.subplot(1, 3, 3)
# plt.title('Repaired Data')
# plt.scatter(data_repaired['feature1'], data_repaired['feature2'])
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
#
# plt.tight_layout()
# plt.show()

# section 变分自编码器VAE，通过引入概率建模，使自编码器能够从异常值中恢复更符合数据分布的重构数据。（需要修改在干净数据上训练自编码器）

# # 数据标准化
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data)
#
# # VAE模型定义
# input_dim = data_scaled.shape[1]
# intermediate_dim = 2
# latent_dim = 2
#
# # 编码器
# inputs = Input(shape=(input_dim,))
# h = Dense(intermediate_dim, activation='relu')(inputs)
# z_mean = Dense(latent_dim)(h)
# z_log_var = Dense(latent_dim)(h)
#
# # 采样层
# def sampling(args):
#     z_mean, z_log_var = args
#     batch = K.shape(z_mean)[0]
#     dim = K.int_shape(z_mean)[1]
#     epsilon = K.random_normal(shape=(batch, dim))
#     return z_mean + K.exp(0.5 * z_log_var) * epsilon
#
# z = Lambda(sampling)([z_mean, z_log_var])
#
# # 解码器
# decoder_h = Dense(intermediate_dim, activation='relu')
# decoder_mean = Dense(input_dim, activation='sigmoid')
# h_decoded = decoder_h(z)
# x_decoded_mean = decoder_mean(h_decoded)
#
# vae = Model(inputs, x_decoded_mean)
#
# # 损失函数
# xent_loss = binary_crossentropy(inputs, x_decoded_mean) * input_dim
# kl_loss = - 0.5 * K.mean(z_log_var - K.square(z_mean) - K.exp(z_log_var) + 1, axis=-1)
# vae_loss = K.mean(xent_loss + kl_loss)
# vae.add_loss(vae_loss)
# vae.compile(optimizer='adam')
#
# # 训练
# vae.fit(data_scaled, epochs=100, batch_size=2, shuffle=True)
#
# # 修复异常值
# reconstructed_data = vae.predict(data_scaled)
# data_repaired = scaler.inverse_transform(reconstructed_data)
# data_repaired = pd.DataFrame(data_repaired, columns=data.columns)
#
# print("Repaired Data using VAE:")
# print(data_repaired)

# section 生成对抗自编码器GAN-AE,实现GAN-AE需要实现生成器和判别器，这通常比VAE复杂得多（需要修改在干净数据上训练自编码器）

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 使用鲁棒标准化
# scaler = RobustScaler()
# data_scaled = scaler.fit_transform(data)

# GAN-AE模型定义
input_dim = data_scaled.shape[1]
latent_dim = 2

# 生成器
def build_generator():
    model = tf.keras.Sequential([
        Dense(16, activation='relu', input_dim=latent_dim),
        Dense(input_dim, activation='sigmoid')
        # Dense(input_dim, activation='linear')
    ])
    return model

# 判别器
def build_discriminator():
    model = tf.keras.Sequential([
        Dense(16, activation='relu', input_dim=input_dim),
        Dense(1, activation='sigmoid')
        # Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    # model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 自编码器（生成器和判别器）
generator = build_generator()
discriminator = build_discriminator()

# 编码器（简单线性映射）
encoder_input = Input(shape=(input_dim,))
encoded = Dense(latent_dim)(encoder_input)
encoded_output = generator(encoded)
autoencoder = Model(encoder_input, encoded_output)

autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')
# autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# 训练自编码器
autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=2, shuffle=True)

# 修复异常值
reconstructed_data = autoencoder.predict(data_scaled)
data_repaired = scaler.inverse_transform(reconstructed_data)
data_repaired = pd.DataFrame(data_repaired, columns=data.columns)

print("Repaired Data using GAN-AE:")
print(data_repaired)