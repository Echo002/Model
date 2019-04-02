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
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集相关的常数设置
INPUT_NODE = 784
OUTPUT_NODE = 10

# 配置神经网络的参数
LAYER1_NODE = 500
# 隐藏层节点数

BATCH_SIZE = 100
# batch中训练数据个数

LEARNING_RATE_BASE = 0.8
# 基础学习率

LEARNING_RATE_DECAY = 0.99
# 学习率的衰减率

REGULARIZATION_RATE = 0.0001
# 复杂正则化中损失函数的系数

TRAINING_STEPS = 30000
# 训练次数

MOVING_AVERAGE_DECAY = 0.99
# 滑动平均衰减率

# 辅助函数，计算神经网络的前向传播结果
def interface(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 没有滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev='0.1'))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev='0.1'))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算前向传播的结果
    y = interface(x, None, weight1, biases1, weight2, biases2)

    # 定义存储轮询的变量
    global_step = tf.Variable(0, trainable=False)

    # 定义滑动平均衰减率和训练轮数的变量
    variable_average = tf.tran.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数
    variable_average_op = variable_average.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后前向传播的结果
    average_y = interface(x, variable_average, weight1, biases1, weight2, biases2)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))

    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算模型的正则化损失
    regularization = regularizer(weight1) + regularizer(weight2)

    # 总损失等于交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + regularization

    # 设置指数学习衰减率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    # 使用tf.train.GradientDescentOptimizer来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 更新神经网络中的参数+参数的滑动平均值
    train_op = tf.group(train_step, variable_average_op)

    # 检验平均滑动模型的前向传播是否正确
    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # 初始化并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.lables}

        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.lables}

        # 迭代训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 100 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print('After %d training step(s),validation accuracy using average model is %g, test accuracy using average model is %g' % (i, validate_acc, test_acc))

            # 产生新的batch数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict=test_feed)

        # 训练结束后，测试正确率
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.lables})
        print('After %d training step(s),test accuracy using average model is %g' % (TRAINING_STEPS, test_acc))


def main(argv=None):
    # 声明数据集的类
    mnist = input_data.read_data_sets('E:\DataSet\MNIST', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()