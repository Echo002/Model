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
import tensorflow as tf
import numpy as np

# 模型目录
CHECKPOINT_DIR = r'E:\Code\DeepLearning\Algorithm\NeuralNetwork\Transfer Learning\Inception_v3\runs\1543994959\checkpoints'
INCEPTION_MODEL_FILE = r'E:\Code\DeepLearning\Algorithm\NeuralNetwork\Transfer Learning\Inception_v3\DataSet\tensorflow_inception_graph.pb'

# inception-v3模型参数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

# 测试数据
file_path = r'E:\Code\DeepLearning\Algorithm\NeuralNetwork\Transfer Learning\Inception_v3\1.jpg'
y_test = [4]

# 读取数据
image_data = tf.gfile.FastGFile(file_path, 'rb').read()

# 评估
checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
with tf.Graph().as_default() as graph:
    with tf.Session().as_default() as sess:
        # 读取训练好的inception-v3模型
        with tf.gfile.FastGFile(INCEPTION_MODEL_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
            graph_def,
            return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

        # 使用inception-v3处理图片获取特征向量
        bottleneck_values = sess.run(bottleneck_tensor,
                                     {jpeg_data_tensor: image_data})
        # 将四维数组压缩成一维数组，由于全连接层输入时有batch的维度，所以用列表作为输入
        bottleneck_values = [np.squeeze(bottleneck_values)]

        # 加载元图和变量
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 通过名字从图中获取输入占位符
        input_x = graph.get_operation_by_name(
            'BottleneckInputPlaceholder').outputs[0]

        # 我们想要评估的tensors
        predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[
            0]

        # 收集预测值
        all_predictions = sess.run(predictions, {input_x: bottleneck_values})

# 如果提供了标签则打印正确率
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print(r'Total number of test examples: {}'.format(len(y_test)))
    print(r'Accuracy: {:g}'.format(correct_predictions / float(len(y_test))))
