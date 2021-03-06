import torch
from torch.autograd import Variable
import math
from PIL import Image
import numpy as np
from .box_utils import nms, _preprocess


def run_first_stage(image, net, scale, threshold):
    """Run P-Net, generate bounding boxes, and do NMS.

    Arguments:
        image: an instance of PIL.Image.
        net: an instance of pytorch's nn.Module, P-Net.
        scale: a float number,
            scale width and height of the image by this number.
        threshold: a float number,
            threshold on the probability of a face when generating
            bounding boxes from predictions of the net.

    Returns:
        a float numpy array of shape [n_boxes, 9],
            bounding boxes with scores and offsets (4 + 1 + 4).
    """

    # scale the image and convert it to a float array
    width, height = image.size
    # print(width, height) == (615, 407)
    # math.ceil对浮点数向上取整
    sw, sh = math.ceil(width*scale), math.ceil(height*scale)
    # print(sw, sh) == (492, 326)
    img = image.resize((sw, sh), Image.BILINEAR)
    img = np.asarray(img, 'float32')

    with torch.no_grad():
        img = Variable(torch.FloatTensor(_preprocess(img)))
    # print("img:",img.shape) == (1,3,326,492)
    output = net(img)
    # print(output[0].data.numpy().shape) == (1,4,158,241)
    # print(output[1].data.numpy().shape) == (1,2,158,241)
    probs = output[1].data.numpy()[0, 1, :, :]
    # print(probs.shape) == (185,241)
    offsets = output[0].data.numpy()
    # print(offsets.shape) == (1,4,158,241)
    # probs: 每个滑动窗口是面部的概率
    # offsets: 转换为真正的边界框

    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return None

    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]


def _generate_bboxes(probs, offsets, scale, threshold):
    """Generate bounding boxes at places
    where there is probably a face.

    Arguments:
        probs: a float numpy array of shape [n, m].
        offsets: a float numpy array of shape [1, 4, n, m].
        scale: a float number,
            width and height of the image were scaled by this number.
        threshold: a float number.

    Returns:
        a float numpy array of shape [n_boxes, 9]
    """

    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    inds = np.where(probs > threshold)
    # python的广播机制
    # np.where:输出满足条件(即非0)元素的下标
    # print(inds)

    if inds[0].size == 0:
        return np.array([])

    # 转换为边界框
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    bounding_boxes = np.vstack([
        np.round((stride*inds[1] + 1.0)/scale),
        np.round((stride*inds[0] + 1.0)/scale),
        np.round((stride*inds[1] + 1.0 + cell_size)/scale),
        np.round((stride*inds[0] + 1.0 + cell_size)/scale),
        score, offsets
    ])
    # why one is added?

    return bounding_boxes.T
