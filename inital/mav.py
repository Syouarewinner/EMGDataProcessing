# %%
import readtxt
import numpy as np


def rectify(data):      # 整流函数
    out = []
    for i in data:
        if i < 125:
            out.append(125 - i)
        else:
            out.append(i - 125)
    return out


def Mav(EMGList):
    channels = len(EMGList)
    length = len(EMGList[0])    # 数据长度
    width = 200                 # 窗口宽度
    step = 50                   # 步长

    # 整流部分
    for i in range(0, channels):
        EMGList[i] = rectify(EMGList[i])
    # 测试print(EMGList[0][200:500])

    # 此处可以改为滑动窗口算法
    out = []
    for i in range(0, channels):
        out.append([])

    for i in range(0, channels):
        for j in range(0, length, step):
            Data = np.array(EMGList[i][j:j+width])
            out[i].append(np.mean(Data))

    return out

# %%


if __name__ == '__main__':
    filename = "E:/txt/Desktop/file/task/20210803 main/data/静息.txt"
    EMGList = readtxt.ReadTxt(filename)[0:6]
    out = Mav(EMGList)
    print(out)

# %%
