# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:37:42 2020
之前一个文件为预处理文件。
用于创造基本的文件排版和数据分类。
主要是基于系统的文件处理工作。

该py文件用于基于VGG的特征提取分类。
@author: 23288
"""
import matplotlib.pyplot as plt
from keras import optimizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from keras.applications import VGG16
import keras

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

base_dir = 'G:\\MachineLearning\\ProcessedPit'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 5
# 确保取证。（吧，我猜的。）
# 这样子iteration会多一些，但不影响epoch。


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    # 承接VGG的输出矩阵，最后返回的feature也是这个形状，共512个通道。
    labels = np.zeros(shape=(sample_count))
    # 为长条形。
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        # inputs是150*150*3*Sample，输出是4*4*512
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        # feature相当于对应位置转换为处理过的特征。
        # 其下标表示的是样本的序数而已，即第a个样本到第b的样本的变换。
        # labels则...嗯，没变，就是拉平了。
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
        # 直到超出提供的数据容量为止，避免重复图像被载入。
    return features, labels
# 最后返回的是铺平flatten的张量。


train_features, train_labels = extract_features(train_dir, 750)
# 共350+400个训练数据。
validation_features, validation_labels = extract_features(validation_dir, 300)
# 共100+200个检验数据。
test_features, test_labels = extract_features(test_dir, 250)
# 共50+200个验证数据。


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
# 到这边模型就建立完成了。


# 这边则是进行判断。

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
