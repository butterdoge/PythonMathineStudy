# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:12:14 2020
路透社新闻分类。
对新闻类别的主题进行多分类。
@author: 23288
"""

import keras
keras.__version__

from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
#从keras内置的数据集中引入数据。
#控制最大单词数目，选取频率前一万的。
#分别装入测试数据和训练数据中，相当于返回了...两个1×2的变量数组。


#此时数据是 Object的数组，因为每个Object的长度不同，所以并不是数组的数组。
#这些Object是单词的有序序列。
#但不满足最后需求的形式，最后要求的是numpy的一个矩阵。
#该数组的下标反映了一共有 多少个 样本。


word_index = reuters.get_word_index()
#是一个 单词 和 序号 的 字典，其中 键 是 单词 ，值 是 数字。
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#反转词典 键值对 关系 的方法。
#要延后三位。
#前三位都是符号位。（？没懂）
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
#对于里面的每一个索引序号，都查询词典里对应的单词并且用“ ”拼接在一起。
#这拼出来的啥玩意。
#总之可以还原，但重要的是知道怎么转换。



#preparing the data


import numpy as np
#之前没用到numpy吗？为什么还需要有引入。

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        #利用了sequences的信息，sequence即是一个对象 的 数组。
        results[i, sequence] = 1.
        #直接将一个数组传入作为下标，好家伙。
        #好家伙，神仙语法。
        #enumerate相当于为每一个样本加序号，可以额外传参设置序号起始位置。
    return results
#批量生成了数组的数组，每一个数组代表一个样本。
#每一个样本数组展现了该样本存在哪些单词。（是否存在）
#i是样本的序号，sequence是样本内部的信息。

#总之生成了这么一个numpy数组，每一行为一个样本。
#每行长度固定。

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)
#将现有的数据张量化。




#这一段在造轮子，可以省略。

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    #这个描述了其形状的定义。
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
#也是一种张量化，生成维度 样本数×维度 的张量。
#每行有唯一一个‘1’，表示对其的判定。
#仍然是一个样本一行，每行长度固定。

# Our vectorized training labels
one_hot_train_labels = to_one_hot(train_labels)
# Our vectorized test labels
one_hot_test_labels = to_one_hot(test_labels)

#下面这个函数和上面定义的一模一样，相当于内置好的one_hot_test.
#轮子造完了。




#这是已经造好的轮子。
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

from keras import models
from keras import layers


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
#因为要区分的类别有46位，传入的元一次有10000个元，所以要尽量保存较多的信息。
model.add(layers.Dense(64, activation='relu'))
#同理，所以再加一层有屌用。


model.add(layers.Dense(46, activation='softmax'))
#起始层是一个二维向量，每一个行代表一个样本，列代表信息，相当于每个对象有n个元，根据这n个元判断。
#每次输入的时候都是一行一行输入。
#输出的labels结果也是一行一行使用。
#最后输出则是一个只有46个输出的样本。

#softmax会使得输出的总和为1.

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              #分类交叉熵用于多分类问题。
              metrics=['accuracy'])
#这个accuracy是用来做什么的来着。

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
#将训练集 留出 1000个 样本 作为 验证集。

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
#将所有的数据输入20轮，一次读写量为512个数据，用这512个数据的值作修正。
#iteration就是用总数除以batch_size
#设置验证函数，但这里的验证函数参不参与修补？

#执行函数，进行拟合测试，进行20次循环。
#一组传入512个数据（？有啥用）

import matplotlib.pyplot as plt

loss = history.history['loss']
#表示每一轮epoch内的损失值，
val_loss = history.history['val_loss']
#表示用测试集得到的每一轮的损失。
#得到的是一个数组，长度是20，每轮对应一个损失。

#损失函数 是 一个 batch内所有数据 代入后的 损失的平均。
#随着epoch的增加，由过拟合变成欠拟合。

#history得到的是一个词典。
#里面含有每一轮epoch里。
#损失的大小和精度的大小。
#用于反映是过拟合还是欠拟合。


epochs = range(1, len(loss) + 1)
#迭代次数。

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#用图像来反映迭代结果。

plt.show()

plt.clf()   # clear figure

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#这边是绘制精度的函数。
plt.show()
#最后通过调整epoch，避免过拟合和欠拟合的问题！好家伙！

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))

#model.evaluate用于评估该模型的精度，而非用于预测一个样本的结果。
results = model.evaluate(x_test, one_hot_test_labels)
#调整后得到一个较为理想化的模型，但不否认是否还有优化空间。
#batch主要用来决定内存的使用情况。


#接下来开始进行预测了。
predictions = model.predict(x_test)
predictions[0].shape
np.sum(predictions[0])

import copy

#拷贝版块。
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
#获得一份拷贝后打乱顺序
float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels)
#寻找对应位置仍然相同的元素个数。

#这个好像用于判断完全随机情况下的数据命中率，来和训练过的模型进行比较判断。

#大概这样，但是如何用于加在数据身上，将这个模型导出验证呢？