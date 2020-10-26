# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:19:55 2020
猫和狗数据比较。
@author: 23288
"""

import keras
import os,shutil


# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = 'G:/CatsVerseDogs/dogs-vs-cats/train/train'
#这边装载了所有的训练数据。

base_dir = 'G:/CatsVerseDogs/dcsmall'
#os.mkdir(base_dir)
#这边装载了之后处理后的小图片数据。

# Directories for our training,
# validation and test splits
#用于训练时的文件目录。
train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)
#拼接得到一个新的目录名。
#创造该目录。
#该目录用于装载过会要用的图像文件。


# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
#os.mkdir(train_cats_dir)

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
#os.mkdir(train_dogs_dir)

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
#os.mkdir(validation_cats_dir)

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
#os.mkdir(validation_dogs_dir)

# Directory with our validation cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
#os.mkdir(test_cats_dir)

# Directory with our validation dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
#os.mkdir(test_dogs_dir)

#准备好容器来装过会要用的文件。
#哎，好累。


# Copy first 1000 cat images to train_cats_dir
#创造一个小动物图片的名字数组装填所有的图片名称。
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
#括号内即为这个数字。
#林子大了什么函数都有。
#for fname in fnames:
#    src = os.path.join(original_dataset_dir, fname)
 #   #创造其所在位置的url
 #   dst = os.path.join(train_cats_dir, fname)
 #   #创造其训练时对应的路径。
 #   shutil.copyfile(src, dst)
 #   #进行复制拷贝。
  #  #拷贝1000个。

# Copy next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
#   shutil.copyfile(src, dst)
    #这500个作为测试集。
    
# Copy next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
 #   shutil.copyfile(src, dst)
    #这500个作为验证集。
    
# Copy first 1000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
 #   shutil.copyfile(src, dst)
    #然后是狗狗。
    
# Copy next 500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
#    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
#   shutil.copyfile(src, dst)
    
print('total training cat images:', len(os.listdir(train_cats_dir)))
    
from keras import layers
from keras import models
#创造他妈的卷积神经网络。

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
#好家伙好家伙好家伙。我就是想写注释，烦死了。
model.add(layers.Flatten())
#降维一定程度后可以扁平化。
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
#输出最后一个判定值（0~1）

from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              #这个是啥来着（？）
              metrics=['acc'])
#设置其优化参数。

from keras.preprocessing.image import ImageDataGenerator
#这个模块内的这个类可以将硬盘上的图像文件自动转换成预处理好的张量批量。
#包括像素值缩放，像素网格转换为浮点数张量，JEPG解码为RGB像素网格。
#获取图像文件等一系列操作。

# All images will be rescaled by 1./255
#将其像素值缩小到0~1区间内。
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        #目标目录。
        # All images will be resized to 150x150
        target_size=(150, 150),
        #目标像素尺寸。
        batch_size=20,
        #每次批量。
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
#损失标签：我也不知道这是个啥。

#同理，从验证集获取数据。
#这时并无法将其和实际的分离：即这里面既包括猫猫也包括狗狗，好耶。
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

#每次生成了一个RBG图像和二进制标签。
#利用生成器，对数据进行拟合。
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

#基于生成器的模型训练。
history = model.fit_generator(
      train_generator,
      #提供一个生成器。
      steps_per_epoch=100,#每一轮抽取多少批。（2000/20=100）
      epochs=30,#执行多少个大轮。
      validation_data=validation_generator,#提供一个验证数据。
      validation_steps=50)
#他是怎么细化出内部是猫还是狗的?

model.save('cats_and_dogs_small_1.h5')
#这玩意保存到哪边了于是乎。

import matplotlib.pyplot as plt

#history中仍然具备所需数据。
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

#画了两张图，一个figure一个show有什么差别？
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()




#设置数据增强来避免过拟合。
datagen = ImageDataGenerator(
      rotation_range=40,#旋转角度
      width_shift_range=0.2,#宽度偏移
      height_shift_range=0.2,#高度偏移。
      shear_range=0.2,#图像切变
      zoom_range=0.2,#
      horizontal_flip=True,
      fill_mode='nearest')

# This is module with image preprocessing utilities
from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# We pick one image to "augment"
img_path = fnames[3]

# Read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))

# Convert it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# The .flow() command below generates batches of randomly transformed images.
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
#相当于将之前缩减255分之一的函数增强了，提供了更多元的功能。

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)
#验证图像没必要进行较多处理。

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)

model.save('cats_and_dogs_small_2.h5')

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