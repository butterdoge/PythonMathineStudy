# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:59:08 2020
垃圾Python第三个代码
@author: 23288
"""

#感觉就是用同一年的指标预测同一年的结果，和时间序列没关系呀。
#好吧，是标量回归，错怪了（
import keras
#日常引入变量。

from keras.datasets import boston_housing
#好像是你妈的预测房价。
(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()
#这边是波斯顿506个月内部的房价指数，首先如下

#这是一个时间序列相关的预测。
#数据量较少，且量纲不同。
#输出结果是一个价格，价格是唯一被预测量。

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std
#数据的标准化处理

#应该是一连窜数据对一个结果的映射。
#这边enmmmm
#？我草，那么直接。
from keras import models
from keras import layers

def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    #每次构造层的时候必须用这个来开始作为第一层。
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    #两个中间层。
    model.add(layers.Dense(1))
    #最后输出是一个数字。
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
#预测函数，使用metrics作为一个算子。
#设置评估指标。（优化的时候是以loss为优化目标，metrics是用来？

import numpy as np
#用k层迭代循环提高数据精度。
k = 4
num_val_samples = len(train_data) // k
#将样本等分为4份。
num_epochs = 100
all_scores = []
#外部变量，每一次迭代后不会重置。
for i in range(k):
    #进行k折运算
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    #选取某一个区间为验证集
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    #同理，y值选择。
    #注意y值就是数据值其后一年的数据。
    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    #选择其他部分作为训练集，concatenate可以将两个矩阵进行合并，意为串联。
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    #调用了一个快速建立模型的函数。
    #建立模型。（?)
    #模型需要动态建立，每次换一批。
    # Train the model (in silent mode, verbose=0)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    #每次拟合采用相同的batch。
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    #主要在于用了不同的数据。
    #在验证集上对模型结果进行评估验证。
    #evaluate返回的是之前设置的matrics和loss函数。
    
    all_scores.append(val_mae)
    #追加，表示在原来数组基础上添加一个元素，这样可以累加k折内的每一次数据。
    
    np.mean(all_scores)
    #取平均值，取完平均值然后可以得知总体的情况？

from keras import backend as K
#计算下一轮，清空处理。
# Some memory clean-up
K.clear_session()
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    #这一块都是准备数据。


    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    #调用history，可以看到每一轮epoch后得到的结果，这里的结果包括metrics和loss函数。
    #然后就可以得到历史记录的一个统合。
    all_mae_histories.append(mae_history)