"""
预备是进行安全帽的图像识别检测。
@author：林金伟
 """


from keras import optimizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import shutil
from keras.applications import VGG16
# 引入训练好的框架。

original_dataset_dir = 'G:\MachineLearning'
# 这边保存原先装好的安全帽和人的数据图。

pathhat = 'G:\MachineLearning\SafeHat'
pathperson = 'G:\MachineLearning\Person'
# 存放安全帽和人像的文件夹。

base_dir = 'G:\MachineLearning\ProcessedPit'
# 这边则是保存处理后的数据图

train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)
# 训练集，测试集，检验集的数据文件夹创建。

train_hats_dir = os.path.join(train_dir, 'hats')
#os.mkdir(train_hats_dir)
train_persons_dir = os.path.join(train_dir, 'persons')
#os.mkdir(train_persons_dir)
# 在训练集内，创建安全帽和普通人的分类文件夹。

validation_hats_dir = os.path.join(validation_dir, 'hats')
#os.mkdir(validation_hats_dir)
validation_persons_dir = os.path.join(validation_dir, 'persons')
#os.mkdir(validation_persons_dir)
# 在检验集内，创建安全帽和普通人的分类文件夹。

test_hats_dir = os.path.join(test_dir, 'hats')
#os.mkdir(test_hats_dir)
test_persons_dir = os.path.join(test_dir, 'persons')
#os.mkdir(test_persons_dir)
# 创建 检验集 部分的分类文件夹。

fnames = ['{}.jpg'.format(i) for i in range(350)]
# 安全帽的名称集。
for fname in fnames:
    src = os.path.join(pathhat, fname)
    dst = os.path.join(train_hats_dir, fname)
    shutil.copyfile(src, dst)
# 考虑到只有500个样本,取350个作为训练集。


fnames = ['{}.jpg'.format(i) for i in range(350, 450)]
for fname in fnames:
    src = os.path.join(pathhat, fname)
    dst = os.path.join(test_hats_dir, fname)
    shutil.copyfile(src, dst)
    # 取100个作为帽子的测试集。

fnames = ['{}.jpg'.format(i) for i in range(450, 500)]
for fname in fnames:
    src = os.path.join(pathhat, fname)
    dst = os.path.join(validation_hats_dir, fname)
    shutil.copyfile(src, dst)
    # 取50个作为验证集


# 之后是person的部分。

fnames = ['{}.jpg'.format(i) for i in range(400)]
for fname in fnames:
    src = os.path.join(pathperson, fname)
    dst = os.path.join(train_persons_dir, fname)
    shutil.copyfile(src, dst)
    # persons共有800个集合，分为400 200 200

# Copy next 500 dog images to validation_dogs_dir
fnames = ['{}.jpg'.format(i) for i in range(400, 600)]
for fname in fnames:
    src = os.path.join(pathperson, fname)
    dst = os.path.join(validation_persons_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 dog images to test_dogs_dir
fnames = ['{}.jpg'.format(i) for i in range(600, 800)]
for fname in fnames:
    src = os.path.join(pathperson, fname)
    dst = os.path.join(test_persons_dir, fname)
    shutil.copyfile(src, dst)

print('total training hat images:', len(os.listdir(train_hats_dir)))

# 进行图像预处理，同时创造一个生成器。
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# 尺寸缩小二百五十五分之一。
train_generator = train_datagen.flow_from_directory(
    train_dir,
    # 目标目录。
    target_size=(150, 150),
    # 目标像素尺寸。
    batch_size=20,
    # 每次批量。
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
#进行特征提取，利用现有的比较完备的VGG16模型。

