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
import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 数据集目录
INPUT_DATA = r'E:\Code\DeepLearning\Algorithm\NeuralNetwork\Transfer Learning\Inception_v3\DataSet'
OUTPUT_FILE = r'E:\Code\DeepLearning\Algorithm\NeuralNetwork\Transfer Learning\Inception_v3\DataSet\flower_processed_data.npy'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def creat_image_lists(sess, testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    # 初始化各个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # 读取子目录
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPGE']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

    # 处理图片数据
    for file_name in file_list:
        # 将图片转化为229*229
        image_raw_data = gfile.FastGFile(file_name, 'rb').read()
        image = tf.image.decode_jpeg(image_raw_data)
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [299, 299])
        image_value = sess.run(image)

        # 随机划分数据集
        chance = np.random.randint(100)
        if chance < validation_percentage:
            validation_images.append(image_value)
            validation_labels.append(current_label)
        elif chance < (testing_percentage + validation_percentage):
            testing_images.append(image_value)
            testing_labels.append(current_label)
        else:
            training_images.append(image_value)
            training_labels.append(current_label)
        current_label += 1

    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return np.asarray(
        [training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels])


def main():
    with tf.Session() as sess:
        processed_data = creat_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        np.save(OUTPUT_FILE, processed_data)

if __name__ == '__main__':
    main()