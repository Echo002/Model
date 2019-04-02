import keras
from keras_applications.imagenet_utils import _obtain_input_shape
# 确定适当的输入形状，相当于opencv中的read.img，将图像变为数组
from keras import backend as K
from keras.layers import Input, Convolution2D, SeparableConv2D,\
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation
from keras.models import Model
from keras.engine.topology import get_source_inputs
# from depthwise_conv2d import DepthwiseConvolution2D
from keras.utils import plot_model

def MobileNet(input_tensor=None, input_shape=None, alpha=1, shallow=True, classes=10):
    """
    # 参数说明
            input_tensor: 输入的tensor，如果不是Keras支持的格式也可以进行转换
            input_shape: 输入的tensor的格式
            alpha: 对应paper中的第一个超参数，用于在深度可分离的卷集中按比例减少通道数
            shallow: 论文中可选的5个stride=1的深度可分离卷积
            classes: 需要分类数
        # Returns
            返回一个Keras model实例

        """

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=28,
                                      data_format=K.image_data_format(),
                                      require_flatten=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(int(64 * alpha), (3, 3), strides=(1, 1), depth_multiplier=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    x = SeparableConv2D(int(128 * alpha), (3, 3), strides=(2, 2), depth_multiplier=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    #
    x = SeparableConv2D(int(128 * alpha), (3, 3), strides=(1, 1), depth_multiplier=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    #
    x = SeparableConv2D(int(256 * alpha), (3, 3), strides=(2, 2), depth_multiplier=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    #
    x = SeparableConv2D(int(256 * alpha), (3, 3), strides=(1, 1), depth_multiplier=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    #
    x = SeparableConv2D(int(512 * alpha), (3, 3), strides=(2, 2), depth_multiplier=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    #
    if not shallow:
        for _ in range(5):
            x = SeparableConv2D(int(512 * alpha), (3, 3), strides=(1, 1), depth_multiplier=1, padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            # x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
            # x = BatchNormalization()(x)
            # x = Activation('relu')(x)
    #
    x = SeparableConv2D(int(1024 * alpha), (3, 3), strides=(2, 2), depth_multiplier=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    #
    x = SeparableConv2D(int(1024 * alpha), (3, 3), strides=(1, 1), depth_multiplier=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)

    out = Dense(classes, activation='softmax')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, out, name='mobilenet')

    return model


if __name__ == '__main__':
    m = MobileNet()
    print(m.summary())
    plot_model(m, to_file='model.png')
    print("model ready")
