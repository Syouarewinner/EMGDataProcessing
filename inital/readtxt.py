# 该文件用于存储读取文件夹，初始化数据函数
# %%
import re
channels = 6


def ReadTxt(FileName):
    """ 读取文件中的数据，并保存到数组中\n
        List = ReadTxt(String)
        List: 返回一个二维数组，第一维为数据长度，第二维为通道数
        String: 文件名字符串

        该函数用只读方式打开txt文件，
        并以readline+正则表达式的方式遍历文件
        文件默认如下形式为一行: \
        "数据名:\\t\\d\\t\\d\\t\\d\\t\\d\\t\\d\\t\\d\\t\\d\\t\\d\\n"

        注意: 该函数以以re的名字使用了re|
        目前只设置了针对肌电信号的匹配读取 """
    EMGList = []
    with open(FileName, 'r') as f:
        lines = f.readlines()
        for line in lines:
            [name, data] = line.split("\t", 1)
            if name == "RawEMG:":
                EMG = [int(s)-127.5 for s in re.findall(r'\d+', data)]
                # EMGList.append(EMG[0:channels])
                EMGList.append([EMG[5], 0])
    return EMGList


def ReadTxt0(FileName):
    EMGList = []
    with open(FileName, 'r') as f:
        lines = f.readlines()
        for line in lines:
            EMG = line.split(',')
            x = range(0, len(EMG))
            for i in x:
                EMG[i] = float(EMG[i])
            EMGList.append(EMG[0:10])
    return EMGList


# %%
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np
    filename = "../data/静息.txt"
    a = np.array(ReadTxt(filename))
    length = len(a)
    x = range(0, length)
    for i in range(0, channels):
        plt.subplot(channels, 1, i+1)
        plt.plot(x, a[..., i])
        plt.yticks([-127.5, 127.5])
# %%
