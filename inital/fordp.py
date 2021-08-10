# %%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm as sk_svm
from sklearn.metrics import accuracy_score
# import pandas as pd
# import feature
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

EMGList = np.array(readtxt.ReadTxt0("../data/6-21.txt"))

dataset = [EMGList[0:10000, 0:8], EMGList[0:10000, 9]]
print(len(dataset[0]), len(dataset[1]))
plot(dataset[0])
plt.plot(dataset[1])
# EMGMAV = MAV(EMGList)

# %%

X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.1)

# %%
# 将EMGMAV用于训练SVM
model = sk_svm.SVC(C=0.5, kernel='rbf', gamma='auto')
model.fit(X_train, y_train)

# %%
# 使用训练好的SVM测试EMG
y_return = model.predict(X_test)
print('R^2: ', model.score(X_test, y_test))
print('准确率: ', accuracy_score(y_test, y_return))

# %%
x = range(0, len(y_test))
y_return = labelsort(y_return, y_test)
y_test = np.sort(y_test)
plt.figure(1, figsize=[50, 50], dpi=20)
plt.plot(x, y_test, color='red', marker='*', label='origin', linestyle='-', linewidth=2)
plt.plot(x, y_return, color='blue', linestyle=':', marker='o',
         label='return', linewidth=1)
plt.show()
# %%
