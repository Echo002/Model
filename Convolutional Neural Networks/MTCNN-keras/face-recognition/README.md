# 人脸检测 (face-recognition)
Face recognition using MTCNN and FaceNet

这个专案主要是用来帮助学习如何使用mtcnn与Google's facenet来设计一个简单易用的人脸辨识系统。

## 参考
* [OpenFace](https://github.com/cmusatyalab/openface)
* Facenet 专案 [davidsandberg](https://github.com/davidsandberg/facenet)
* [bearsprogrammer](https://github.com/bearsprogrammer/real-time-deep-face-recognition)
* [shanren7](https://github.com/shanren7/real_time_face_recognition)

## 依赖函数库
* Tensorflow 1.2.1
* Python 3.5
* 以及在[davidsandberg](https://github.com/davidsandberg/facenet)专案的[requirement.txt](https://github.com/davidsandberg/facenet/blob/master/requirements.txt)要求

## 预先训练的模型
* Inception_ResNet_v1 CASIA-WebFace-> [20170511-185253](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE/edit)

## 使用MTCNN进行人脸对齐
* 你需要从[davidsandberg](https://github.com/davidsandberg/facenet)的专案中下载[det1.npy, det2.npy, and det3.npy](https://github.com/davidsandberg/facenet/tree/master/src/align) 

