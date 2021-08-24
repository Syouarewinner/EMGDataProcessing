# 该文件用于存储特征提取函数
# %%
import pandas as pd
import math
import numpy as np


def rms(x):
    # 取一个一维序列的均方根
    length = len(x)
    sum = 0
    for i in x:
        sum += i*i
    return math.sqrt(sum/length)


def feature(EMGList):
    """ 提取输入矩阵的特征值\n
    ndarray = feature(EMGList)
    ndarray: 输入信号的特征值（第一维为滑窗后数据长度，第二维为通道数*特征数）
    EMGList: 输入的信号（第一维为滑窗后数据长度，第二维为通道数）\n
    注意: 该函数使用以pd为名调用了pandas| 以np为名调用了numpy \
        | 调用了math """
    width = 200
    step = 50
    df = pd.DataFrame(EMGList)
    # 在此处添加特征值
    # 利用rolling获得滑动窗口
    r_abs = df.abs().rolling(width, min_periods=width)
    r = df.rolling(width, min_periods=width)
    df_mav = r_abs.mean()[0:-1:step].dropna()
    # df_rms = r_abs.agg(rms)[0:-1:step].dropna()
    df_var = r.var()[0:-1:step].dropna()

    Feature = pd.concat([df_mav
                        # , df_rms
                         , df_var
                         ], axis=1)
    # print(np.array(Feature))
    return np.array(Feature)


if __name__ == '__main__':
    import readtxt
    filename = "../data/静息.txt"
    EMGList = readtxt.ReadTxt(filename)
    Feature = feature(EMGList)
    print(Feature)
# %%
