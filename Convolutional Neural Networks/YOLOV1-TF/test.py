import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer


class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        #self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        print ('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result):
        for i in range(len(result)):  # 没进入循环
            # print(result)
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 1)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            #cv2.rectangle(img, (20, 20), (250, 250), (255, 0, 0), 1)
            #cv2.imshow('Image', img)
            # cv2.waitKey()
            # cv2.rectangle(img,)
            cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))   # 调整统一大小
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)  # 转为浮点型，为什么数值全为255？
        inputs = (inputs / 255.0) * 2.0 - 1.0   # 像素值归一化
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))   # 1X7x7x3
        print('inputs=%s' % inputs)

        result = self.detect_from_cvmat(inputs)[0]    # 进入角点检测results[0]
        print(result)

        for i in range(len(result)):  # 没进入循环
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})  # 将图片输入网络
        results = []   # 定义了一个存放结果的list
        for i in range(net_output.shape[0]):    # 是为了判断输出是视频还是一张图片？？？
            results.append(self.interpret_output(net_output[i]))
        # c = net_output[0]      # 1x9900大小
        #print('net_output.shape[0]:%s' % net_output.shape[0])
        # b = results   # results长度也就为1(似乎一张图片就为1)
        return results

    # 输出预测结果
    def interpret_output(self, output):   # 处理网络输出的1x9900个数据
        probs = np.zeros((self.cell_size, self.cell_size, self.boxes_per_cell, self.num_class))  # 定义了概率

        # 一下三项包含了全部7x7x11个数据
        class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))     # 7x7x1 物体属于哪一类的概率
        scales = np.reshape(output[self.boundary1:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))   # 7x7x2的预测框
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))           # 7x7x2x4个坐标信息
        print('class_probs:%s\n scales:%s\n boxes:%s' % (class_probs, scales, boxes))
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),  # 两个偏差
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))
        # 似乎只是将原来的像素信息中心化后的值，正负都有
        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])   # 平方后变正

        boxes *= self.image_size    # 坐标信息反归一化 boxes的w和h都偏小

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(      # 置信度
                    class_probs[:, :, j], scales[:, :, i])   # class概率对，但scales概率太小了，整体结果很小
        # 没有检测到角点
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')  # self.threshold=0.2,返回值为bool
        filter_mat_boxes = np.nonzero(filter_mat_probs)    # 选出大于阈值的坐标信息
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[
            0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):  # i不在范围内
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold: # self.iou_threshold=0.5
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):   # i不在范围内
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[
                          i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)
        # cv2.imshow('a', image)
        # cv2.rectangle(image, (20, 20), (25, 25), (255, 0, 0), 1)  # 绘制测试图
        # cv2.imshow('a', image)

        detect_timer.tic()
        result = self.detect(image)
        # relen = len(result)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

        self.draw_result(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)



def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weights', default="save.ckpt-19000", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(False)
    # weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    weight_file = os.path.join(args.data_dir, 'pascal_voc', 'output', '2018_01_03_09_17', args.weights)
    detector = Detector(yolo, weight_file)

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from image file
    imname = 'test/000015.jpg'
    # image = cv2.imread(imname)
    # cv2.rectangle(image, (20, 20), (250, 250), (255, 0, 0), 10)
    # cv2.imshow('a',image)
    # cv2.waitKey(0)
    detector.image_detector(imname)


if __name__ == '__main__':
    main()