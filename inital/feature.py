# %%
import pandas as pd
import math
import numpy as np


def rms(x):
    length = len(x)
    sum = 0
    for i in x:
        sum += i*i
    return math.sqrt(sum/length)


def feature(EMGList):
    width = 200
    step = 50
    df = pd.DataFrame(EMGList)
    # 利用rolling获得滑动窗口
    r_abs = df.abs().rolling(width, min_periods=width)
    r = df.rolling(width, min_periods=width)
    df_mav = r_abs.mean()[0:-1:step].dropna()
    df_rms = r_abs.agg(rms)[0:-1:step].dropna()
    df_var = r.var()[0:-1:step].dropna()

    Feature = pd.concat([df_mav, df_rms, df_var], axis=1)
    return np.array(Feature)


if __name__ == '__main__':
    import readtxt
    filename = "../data/静息.txt"
    EMGList = readtxt.ReadTxt(filename)
    Feature = feature(EMGList)
    print(Feature)
# %%
