# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 19:45:22 2020

@author: 23288
"""
import tensorflow as tf

from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.compat.v1.Session(config=config)

set_session(sess)


from keras import layers
from keras import models


import tensorflow as tf 
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True 
session = tf.compat.v1.InteractiveSession(config=config)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#用3×3来进行卷积，深度为32层，包含32个特征通道。
#激活函数为relu，取正值，忽略负值。
#输入的形状为...
model.add(layers.MaxPooling2D((2, 2)))
#最大池化，将其浓缩减小一维。
#是什么原理（？）
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#此时有输出为11×11，每个对应深度为64.
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#最后这边输出为3×3的五块，有64个特征通道。
model.add(layers.Flatten())
#铺展开。
model.add(layers.Dense(64, activation='relu'))
#进行9×64至64的映射。
model.add(layers.Dense(10, activation='softmax'))
#用softmax，来保证所有的概率期望加起来为1.
model.summary()
from keras.datasets import mnist
#获取数字识别数据。
from keras.utils import to_categorical
#将label张量化的工具，从1到10变换到1~10的向量。
#为什么不直接输出一个值，结果是0~10？
#因为这样就不会既偏向于2又偏向于4了。
#会和相邻项贴近。
#实际上的预测是有跳跃的。

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#获取数据

train_images = train_images.reshape((60000, 28, 28, 1))
#重新整顿数据。
train_images = train_images.astype('float32') / 255
#为什么要这么做（？）

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#张量化处理。
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#设置编译，采用多分类模型，同时以精度为metrics
model.fit(train_images, train_labels, epochs=5, batch_size=64)
#一次迭代64个数据，进行5大轮。
test_loss, test_acc = model.evaluate(test_images, test_labels)
#得到模型的评估结果。