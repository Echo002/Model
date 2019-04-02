#!/usr/bin/env python
#!-*-coding:utf-8 -*-
#!@Author:xugao
#         ┌─┐       ┌─┐
#      ┌──┘ ┴───────┘ ┴──┐
#      │                 │
#      │                 │
#      │    ＞  　　＜    │
#      │                 │
#      │  ....　⌒　....　│
#      │                 │
#      └───┐         ┌───┘
#          │         │
#          │         │
#          │         │
#          │         └──────────────┐
#          │                        │
#          │                        ├─┐
#          │                        ┌─┘
#          │                        │
#          └─┐  ┐  ┌───────┬──┐  ┌──┘
#            │ ─┤ ─┤       │ ─┤ ─┤
#            └──┴──┘       └──┴──┘
#                神兽保佑
#                BUG是不可能有BUG的!
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mobilenet import MobileNet
import numpy as np
np.random.seed(10)
from keras.datasets import cifar10
from keras.utils import np_utils

(x_img_train, y_label_train),(x_img_test, y_label_test) = cifar10.load_data()

x_img_train = x_img_train.astype('float')/255.0
x_img_test = x_img_test.astype('float')/255.0

y_label_train = np_utils.to_categorical(y_label_train)
y_label_test = np_utils.to_categorical(y_label_test)

model = MobileNet()
try:
    model.load_weights("mobileV1-lite.h5")
    print("模型加载成功！继续训练")
except:
    print("模型加载失败！从头开始训练")

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x_img_train, y_label_train, validation_split=0.2, epochs=10, batch_size=128, verbose=2)
model.save_weights("mobileV1-lite.h5")
print("保存模型成功！")