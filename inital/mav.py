# %%
import readtxt
import numpy as np


def rectify(data, mean=128):      # 整流函数
    """ 将输入的数据进行整流\n
    List = rectify(ListIn,mean)
    List: 输出为一个一维数组
    ListIn: 输入为一个一维数组
    事实上只要能用for循环进行遍历并计算，都可以作为该函数的输入
    mean: 输入数据的零值，默认为适配肌电的128 """
    out = []
    for i in data:
        if i < mean:
            out.append(mean - i)
        else:
            out.append(i - mean)
    return out


def Mav(EMGList, mean=128):
    """ 使用滑动窗口法对数据进行取绝对平均值处理 \
        (输入的数据如果零值不为128，则需要配置mean参数)\n
        ndarry = Mav(EMGList, mean=128)
        ndarry: 输出的值是一个numpy数组，\
            第一维为数据长度，第二维为通道数
        EMGList: 输入的是一个二维数组，格式与输出相同\
            (输入应是一个list或其他可以支持np.array()\
                生成ndarray的类型)
        mean: 数据的零值，常规意义上应为0，\
            但得到的肌电信号零值为128，故默认为适配肌电的128\n
        注意: 该函数以np的名字使用了numpy|调用了rectify函数 """
    width = 200                 # 窗口宽度
    step = 50                   # 步长
    length = len(EMGList)
    channels = len(EMGList[0])     # 数据长度

    inList = np.array(EMGList).T

    out = []
    for i in range(0, channels):
        out.append([])

    # 整流部分
    for i in range(0, channels):
        out[i] = rectify(inList[i, ...], mean)
    # 测试print(EMGList[0][200:500])

    # 此处可以改为滑动窗口算法
    for i in range(0, channels):
        for j in range(0, length, step):
            Data = np.array(out[i][j:j+width])
            out[i].append(np.mean(Data))

    return np.array(out).T
# %%


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    filename = "../data/静息.txt"
    EMGList = readtxt.ReadTxt(filename)
    out = Mav(EMGList)

    length1 = len(EMGList[0])
    x1 = range(0, length1)

    for i in range(1, 7):
        plt.subplot(6, 1, i)
        plt.plot(x1, EMGList[i-1])
        plt.yticks([0, 256])
    plt.show()
    length2 = len(out)
    x2 = range(0, length2)
    for i in range(1, 7):
        plt.subplot(6, 1, i)
        plt.plot(x2, out[..., i-1])
        plt.yticks([0, 125])
    plt.show()

# %%
