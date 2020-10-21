# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:20:54 2020

@author: 23288
"""
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 引入mnist图像库，将其数据装在训练图像，训练标签中。

# 引入模型和层的概念。

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
# 创造序列。
# 设置两个层，第一层执行relu激活函数，后面为输入形状。

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
# 设置优化编译模型，包括梯度优化器，损失函数设置，衡量标准。

train_images = train_images.reshape((60000, 28 * 28))
# 对图片进行降维处理，使得单个图片二维转一维，每个维度对应一个灰度。
train_images = train_images.astype('float32') / 255
# 转换为浮点型并且使得其范围成为0到1的映射。


test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
# 同理操作。


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# 整数数据转换为boolen。

network.fit(train_images, train_labels, epochs=5, batch_size=128)
# 执行，选择训练数据，注意network是变量，相当于对某一个对象执行该训练操作。
# batch_size是...

test_loss, test_acc = network.evaluate(test_images, test_labels)
# 进行模型评估，获得损失值和精度数据。

print('test_acc:', test_acc)
# 输出。
11
