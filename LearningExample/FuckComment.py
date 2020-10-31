# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:04:47 2020

@author: 23288
"""

import keras
keras.__version__
#获得其版本信息。

from keras.datasets import imdb
#引入imdb数据。

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#读入数据。
#数据的格式为对象的数组（数组每一个元素对应一个样本，每一个样本具有一个代表单词基的序列）
#表示只获取前10000高频词汇。

import numpy as np
#引入numpy格式，只有该格式才可以进行相关活动。

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    #sequences是原始数组容量数据。
    results = np.zeros((len(sequences), dimension))
    #获得一个样本容量×单词数的矩阵，每一个样本获得一个单词容量。
    for i, sequence in enumerate(sequences):
        #enumerate对该sequences的每一个样本
        #i是这个样本的索引，sequence则表示每一个序列。
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
        #i是样本的序号，sequence为一个数组对象。
    return results
#定义一个函数，用于将原始数据转换成numpy格式的tensor（张量）
#这边涉及到信息丢失。

# Our vectorized training data
x_train = vectorize_sequences(train_data)
#执行该操作。
# Our vectorized test data
x_test = vectorize_sequences(test_data)
#将信息张量化，最后是一个容量×序列信息的张量。

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
#asarray将任意数组转换为numpy数组。
#同时设置类型。

from keras import models
from keras import layers
#引入神经网络相关。

model = models.Sequential()
#建立一个序列。
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
#输入10000个变量，输出16个变量，激活函数采用relu，也就是取正。
#按理，需要的参数：10000*16,16w个，涉及上b，那么就有16w+16个。
model.add(layers.Dense(16, activation='relu'))
#建立第二个层，自动衔接上一层
model.add(layers.Dense(1, activation='sigmoid'))
#最后输出一个值，采用激活函数，只映射到1个量。
#添加层。

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#优化器默认用这个就好，损失函数用二项交叉，因为是分类问题。


#如果想进一步设置参数的话就用调用以下这些语句。
#from keras import optimizers
#优化器。

#model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#另一种优化模型（？）

#from keras import losses
#from keras import metrics

#model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#              loss=losses.binary_crossentropy,
#             metrics=[metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
#分成两部分数据，训练集和测试集。

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=50,
                    #回调次数。
                    batch_size=512,
                    #每一轮批输入的数目数。
                    validation_data=(x_val, y_val))
#测试集选择，用于每次获取修正参数。

history_dict = history.history
#history变量含有在训练过程中涉及到的参数。
history_dict.keys()
#获取内部有哪些变量。

from keras import backend as K
K.clear_session()
from numba import cuda
cuda.select_device(0)
cuda.close()

import matplotlib.pyplot as plt
#引入画图函数。



from keras import backend as K
import gc
del model
gc.collect()
K.clear_session()
