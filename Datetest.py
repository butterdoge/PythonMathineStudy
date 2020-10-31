import matplotlib.pyplot as plt
from keras import optimizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from keras.applications import VGG16
# 引入训练好的框架。

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
# 设置该模型引入时的参数。
# 因为最后的dense层是由自己创建，所以选择False。
# weights是权重检查点。

# 引入图像处理机。
# 最后一层出来的是一个4,4,512，的卷积矩阵。

base_dir = '/Users/fchollet/Downloads/cats_and_dogs_small'
# 基本的图像目录。

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
# 设置基本目录，位于基本目录周遭。

datagen = ImageDataGenerator(rescale=1./255)
# 利用生成器将尺寸缩小。

batch_size = 20
# 设置循环batch。


def extract_features(directory, sample_count):
    # 投入一个数据目录和一个样本数，输出一个返回结果。
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    # 最后一层的张量。
    labels = np.zeros(shape=(sample_count))
    # 引入最后一层的label。
    generator = datagen.flow_from_directory(
        directory,  # 获取的目录，从这个目录里制造一个生成器。
        target_size=(150, 150),  # 设置图像尺寸。
        batch_size=batch_size,  # 设置单个batch尺寸。
        class_mode='binary')  # 因为是二分类。
    i = 0
    for inputs_batch, labels_batch in generator:  # 对生成的数据进行处理。
        features_batch = conv_base.predict(inputs_batch)
        # 将其输入到现有的网络，最后输出得到卷积矩阵，4×4×512.
        # 将卷积矩阵铺平。
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        # features是一个4×4×512的矩阵。
        # 绑定现在的测试量。
        # 有点没搞懂这个生成机的机理。
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        # 绑定现在的监督量。

        # 最终得到的也是一个张量！

        # 相当于铺展开来了，类似于那个...flatten？
        i += 1
        if i * batch_size >= sample_count:
            # 会循环生成。
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
        # 直到超过了样本容量的时候就停止。
    return features, labels
# 返回特征输出层 和 测试标签。


train_features, train_labels = extract_features(train_dir, 2000)
# 将训练数据投入，返回一个变换后的结果。
validation_features, validation_labels = extract_features(validation_dir, 1000)
# 将检验数据也投入。
test_features, test_labels = extract_features(test_dir, 1000)
# 将验证数据也投入。

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
# 将处理后的数据重新排列维度。
# 原先是一长条的，
# 又重新变成了dence。


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
# 添加几个层。

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
# 进行编译。

history = model.fit(train_features, train_labels,  # 投入训练数据。
                    epochs=30,  # 设置循环次数。
                    batch_size=20,  # 设置batch尺寸。
                    validation_data=(validation_features, validation_labels))
# 获得其fit结果。


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


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# 先建立了一个模型，最后输出0~1.

print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False
# 冻结，不要让权重再被修改。

print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    train_dir,
    # All images will be resized to 150x150
    target_size=(150, 150),
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2)
