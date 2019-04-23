import numpy as np
import tensorflow as tf
import yolo.config as cfg

slim = tf.contrib.slim


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):

    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride = 1, padding = 'SAME', scope=scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total//2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0,0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride = stride, padding = 'SAME', scope = scope)


class YOLONet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES   #类别
        self.num_class = len(self.classes) #类别数量，值为2
        self.image_size = cfg.IMAGE_SIZE  #图像尺寸,值为512
        self.cell_size = cfg.CELL_SIZE #cell尺寸，值为16
        self.boxes_per_cell = cfg.BOXES_PER_CELL #每个grid cell负责的boxes，默认为2
        self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
###############3
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.logits = self.build_network(self.images, num_outputs=self.output_size, alpha=self.alpha, is_training=is_training)  # 网络的7x7x30输出
             # 7X7X11
        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 5+ self.num_class]) # 以None(batch)x7x7x(5+20)形式
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)
##############3

    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=leaky_relu(alpha),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                # net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')  # 填充
                # net0 = conv2d_same(images, 16, 7, padding='SAME', scope='conv_2')
                net1 = conv2d_same(images, 16, 7, stride = 1, scope='conv_1')
                # net2 = slim.max_pool2d(net1, 2, padding='SAME', scope='pool_1')   # 1
                norm1 = slim.batch_norm(net1, scope = 'norm1')

                short_cut1 = slim.conv2d(norm1, 16, 1, stride = 1, scope = 'shortcut1')
                residual11 = conv2d_same(norm1, 16, 3, stride = 1, scope='conv_2')
                residual12 = conv2d_same(residual11, 16, 3, stride = 1, scope='conv_3')
                residual13 = conv2d_same(residual12, 16, 3, stride = 1, scope='conv_4')
                net3 = residual13 + short_cut1
                norm2 = slim.batch_norm(net3, scope='norm2')

                net4 = slim.max_pool2d(norm2, 2, padding='SAME', scope='pool_2')   # 2
                short_cut2 = slim.conv2d(net4, 32, 1, stride=1, scope='shortcut2')
                residual21 = conv2d_same(net4, 32, 3, stride=1, scope='conv_5')
                residual22 = conv2d_same(residual21, 32, 3, stride=1, scope='conv_6')
                residual23 = conv2d_same(residual22, 32, 3, stride=1, scope='conv_7')
                net5 = residual23 + short_cut2
                norm3 = slim.batch_norm(net5, scope='norm3')

                net6 = slim.max_pool2d(norm3, 2, padding='SAME', scope='pool_3')  # 3
                short_cut3 = slim.conv2d(net6, 64, 1, stride=1, scope='shortcut3')
                residual31 = conv2d_same(net6, 64, 3, stride=1, scope='conv_8')
                residual32 = conv2d_same(residual31, 64, 3, stride=1, scope='conv_9')
                residual33 = conv2d_same(residual32, 64, 3, stride=1, scope='conv_10')
                net7 = residual33 + short_cut3
                norm4 = slim.batch_norm(net7, scope='norm4')

                net8 = slim.max_pool2d(norm4, 2, padding='SAME', scope='pool_4')  # 4
                short_cut4 = slim.conv2d(net8, 128, 1, stride=1, scope='shortcut4')
                residual41 = conv2d_same(net8, 128, 3, stride=1, scope='conv_11')
                residual42 = conv2d_same(residual41, 128, 3, stride=1, scope='conv_12')
                residual43 = conv2d_same(residual42, 128, 3, stride=1, scope='conv_13')
                net9 = residual43 + short_cut4
                norm5 = slim.batch_norm(net9, scope='norm5')

                net10 = slim.max_pool2d(norm5, 2, padding='SAME', scope='pool_5')  # 5
                short_cut5 = slim.conv2d(net10, 256, 1, stride=1, scope='shortcut5')
                residual51 = conv2d_same(net10, 256, 3, stride=1, scope='conv_14')
                residual52 = conv2d_same(residual51, 256, 3, stride=1, scope='conv_15')
                residual53 = conv2d_same(residual52, 256, 3, stride=1, scope='conv_16')
                net11 = residual53 + short_cut5
                norm6 = slim.batch_norm(net11, scope='norm6')

                net17 = slim.conv2d(norm6, 512, 3, scope='conv_18')
                net18 = slim.max_pool2d(net17, 2, padding='SAME', scope='pool_1')  # 1
                net19 = tf.transpose(net18, [0, 3, 1, 2], name='trans_31')
                net20 = slim.flatten(net19, scope='flat_32')
                net21 = slim.fully_connected(net20, 1024, scope='fc_33')
                net22 = slim.fully_connected(net21, 2048, scope='fc_34')
                net23 = slim.dropout(net22, keep_prob=keep_prob,
                                   is_training=is_training, scope='dropout_35')
                net24 = slim.fully_connected(net23, num_outputs,
                                           activation_fn=None, scope='fc_36')    # 输出还是一维
        return net24

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):   # 上角点和下角点
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])  # 变为5x30x30x2x4

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # calculate the left up point & right down point(交集)
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])  #
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])  #

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]  # 重合部分面积

            # calculate the boxs1 square and boxs2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)  # box1并box2的面积

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)  # box1交box2/box1并box2(IOU),并限定了0——1这个范围

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            # 前20个表示类别,加上45个batch重新整理为45x7x7x20
            predict_classes = tf.reshape(predicts[:, :self.boundary1], [self.batch_size, self.cell_size, self.cell_size, self.num_class])
                                                     # 接下来的2个同理
            predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(predicts[:, self.boundary2:], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
            # label只有只有6个参数
            # 将真实的  labels 转换为相应的矩阵形式
            response = tf.reshape(labels[:, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, 1])  # label第一个参数45x7x7x1
            boxes = tf.reshape(labels[:, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4]) # label的1到4参数45x7x7x1x4
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size   # 拼接成两个boxes
            classes = labels[:, :, :, 5:]          # label的5到后面参数

            offset = tf.constant(self.offset, dtype=tf.float32)
            offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])  # 变为5x30x30x2的形式
            predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,  # x
                                           (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size, # y
                                           tf.square(predict_boxes[:, :, :, :, 2]),  # w
                                           tf.square(predict_boxes[:, :, :, :, 3])])  # h
            predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])   # 4x5x30x30x2转换为5x30x30x2x4

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)   # 在第三维求有物体时 5x30x30x2
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask   #1-object_mask 无物体时 5x30x30x2

            boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                                   boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                                   tf.sqrt(boxes[:, :, :, :, 2]),
                                   tf.sqrt(boxes[:, :, :, :, 3])])
            boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)   # 将所有的损失部分统一起来
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):
    def op(inputs):
        return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
    return op
