# %%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm as sk_svm
from sklearn.metrics import accuracy_score
# import pandas as pd
import feature
import readtxt


def plot(InList):
    """ 将输入的数组按通道的顺序绘制成图
    plot(InList)
    InList: 输入的二维矩阵，必须是ndarray类型\
        (第一维为数据长度，第二维为通道数)\n
    注意: 该函数以plt为名使用了matplotlib.pyplot """
    channel = len(InList[0])
    x = range(0, len(InList))
    for i in range(channel):
        plt.subplot(channel, 1, i+1)
        plt.plot(x, InList[..., i])
    plt.show()


# %%


def labelsort(inNdarry, label):
    """ 按照标签对数组进行排序，但该方案目前只支持一维\n
    ndarry = labelsort(inNdarry, label)
    ndarry: 返回的是一个一维的numpy数组
    inNdarry: 排序的数组可以是支持np.append的任意一维数据结构，\
        如ndarry和list
    label: 进行排序的标签可以是支持np.argsort的任意一维数据结构，\
        如ndarry和list\n
    注意: 该函数以np为名调用了numpy """
    x = np.argsort(label)
    outList = np.empty([0, 0])
    for i in x:
        outList = np.append(outList, inNdarry[i])
    return outList


# %%
# 读取数据
EMGList = readtxt.ReadTxt("../data/静息.txt")
ndarry = feature.feature(EMGList)
label = np.ones(1000)
dataset = [ndarry[0:1000, ...], label]
# EMGMAV = MAV(EMGList)

# %%
EMGList = readtxt.ReadTxt("../data/内收.txt")
MavList = feature.feature(EMGList)
label = np.ones(1000)*2
dataset[0] = np.row_stack([dataset[0], MavList[0:1000, ...]])
dataset[1] = np.append(dataset[1], label)
print(1)

EMGList = readtxt.ReadTxt("../data/屈腕.txt")
MavList = feature.feature(EMGList)
label = np.ones(1000)*3
dataset[0] = np.row_stack([dataset[0], MavList[0:1000, ...]])
dataset[1] = np.append(dataset[1], label)
print(2)

EMGList = readtxt.ReadTxt("../data/伸腕.txt")
MavList = feature.feature(EMGList)
label = np.ones(1000)*4
dataset[0] = np.row_stack([dataset[0], MavList[0:1000, ...]])
dataset[1] = np.append(dataset[1], label)
print(3)

EMGList = readtxt.ReadTxt("../data/伸掌.txt")
MavList = feature.feature(EMGList)
label = np.ones(1000)*5
dataset[0] = np.row_stack([dataset[0], MavList[0:1000, ...]])
dataset[1] = np.append(dataset[1], label)
print(4)

EMGList = readtxt.ReadTxt("../data/握拳.txt")
MavList = feature.feature(EMGList)
label = np.ones(1000)*6
dataset[0] = np.row_stack([dataset[0], MavList[0:1000, ...]])
dataset[1] = np.append(dataset[1], label)
print(5)

# %%
# 分割训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.3)

# %%
# 将EMGMAV用于训练SVM
model = sk_svm.SVC(C=0.21, kernel='rbf', gamma='auto')
model.fit(X_train, y_train)

# %%
# 使用训练好的SVM测试EMG
y_return = model.predict(X_test)
print('R^2: ', model.score(X_test, y_test))
print('准确率: ', accuracy_score(y_test, y_return))

# %%
# 画出识别曲线
x = range(0, len(y_test))
y_return = labelsort(y_return, y_test)
y_test = np.sort(y_test)
plt.figure(1, figsize=[50, 50], dpi=20)
plt.plot(x, y_test, color='red', marker='*', label='origin', linestyle='-', linewidth=2)
plt.plot(x, y_return, color='blue', linestyle=':', marker='o',
         label='return', linewidth=1)
plt.show()
# %%
