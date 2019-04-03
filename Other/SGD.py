import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as style


# 构造数据
def get_data(sample_num=1000):
    """
    拟合函数为
    y = 5*x1 + 7*x2
    :return:
    """
    x1 = np.linspace(0, 9, sample_num)
    x2 = np.linspace(4, 13, sample_num)
    x = np.concatenate(([x1], [x2]), axis=0).T
    y = np.dot(x, np.array([5, 7]).T)
    return x, y


# 梯度下降法
def SGD(samples, y, step_size=2, max_iter_count=1000):
    """
    :param samples: 样本
    :param y: 结果value
    :param step_size: 每一接迭代的步长
    :param max_iter_count: 最大的迭代次数
    :param batch_size: 随机选取的相对于总样本的大小
    :return:
    """
    # 确定样本数量以及变量的个数初始化theta值

    m, var = samples.shape
    theta = np.zeros(2)
    y = y.flatten()
    # 进入循环内
    loss = 1
    iter_count = 0
    iter_list = []
    loss_list = []
    theta1 = []
    theta2 = []
    # 当损失精度大于0.01且迭代此时小于最大迭代次数时，进行
    while loss > 0.01 and iter_count < max_iter_count:
        loss = 0
        # 梯度计算
        theta1.append(theta[0])
        theta2.append(theta[1])
        # 样本维数下标
        rand1 = np.random.randint(0, m, 1)
        h = np.dot(theta, samples[rand1].T)
        # 关键点，只需要一个样本点来更新权值
        for i in range(len(theta)):
            theta[i] = theta[i] - step_size * (1 / m) * (h - y[rand1]) * samples[rand1, i]
        # 计算总体的损失精度，等于各个样本损失精度之和
        for i in range(m):
            h = np.dot(theta.T, samples[i])
            # 每组样本点损失的精度
            every_loss = (1 / (var * m)) * np.power((h - y[i]), 2)
            loss = loss + every_loss

        print("iter_count: ", iter_count, "the loss:", loss)

        iter_list.append(iter_count)
        loss_list.append(loss)

        iter_count += 1
    plt.plot(iter_list[:10], loss_list[:10])
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.show()
    return theta1, theta2, theta, loss_list


def painter3D(theta1, theta2, loss):
    style.use('MacOSX')
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    x, y, z = theta1, theta2, loss
    ax1.plot_wireframe(x, y, z, rstride=5, cstride=5)
    ax1.set_xlabel("theta1")
    ax1.set_ylabel("theta2")
    ax1.set_zlabel("loss")
    plt.show()


if __name__ == '__main__':
    samples, y = get_data()
    theta1, theta2, theta, loss_list = SGD(samples, y)
    print(theta)  # 会很接近[5, 7]

    # painter3D(theta1, theta2, loss_list)