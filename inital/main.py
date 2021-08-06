# %%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm as sk_svm
from sklearn.metrics import accuracy_score
import readtxt
import mav


def plot(InList):
    """ 将输入的数组按通道的顺序绘制成图
    plot(InList)
    InList: 输入的二维矩阵，必须是ndarray类型\
        (第一维为数据长度，第二维为通道数)\n
    注意: 该函数以plt为名使用了matplotlib.pyplot """
    channel = len(InList[0])
    x = range(0, len(InList))
    for i in range(channel):
        plt.subplot(6, 1, i+1)
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

EMGList = readtxt.ReadTxt("../data/静息.txt")
MavList = mav.Mav(EMGList)
label = np.ones(1000)
dataset = [MavList[1200:2200, ...], label]
# EMGMAV = MAV(EMGList)

EMGList = readtxt.ReadTxt("../data/内收.txt")
MavList = mav.Mav(EMGList)
label = np.ones(1000)*2
dataset[0] = np.row_stack([dataset[0], MavList[50000:51000, ...]])
dataset[1] = np.append(dataset[1], label)

EMGList = readtxt.ReadTxt("../data/屈腕.txt")
MavList = mav.Mav(EMGList)
label = np.ones(1000)*3
dataset[0] = np.row_stack([dataset[0], MavList[30000:31000, ...]])
dataset[1] = np.append(dataset[1], label)

EMGList = readtxt.ReadTxt("../data/伸腕.txt")
MavList = mav.Mav(EMGList)
label = np.ones(1000)*4
dataset[0] = np.row_stack([dataset[0], MavList[30000:31000, ...]])
dataset[1] = np.append(dataset[1], label)

EMGList = readtxt.ReadTxt("../data/伸掌.txt")
MavList = mav.Mav(EMGList)
label = np.ones(1000)*5
dataset[0] = np.row_stack([dataset[0], MavList[50000:51000, ...]])
dataset[1] = np.append(dataset[1], label)

EMGList = readtxt.ReadTxt("../data/握拳.txt")
MavList = mav.Mav(EMGList)
label = np.ones(1000)*6
dataset[0] = np.row_stack([dataset[0], MavList[20000:21000, ...]])
dataset[1] = np.append(dataset[1], label)

# %%

X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.1)

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
x = range(0, len(y_test))
y_return = labelsort(y_return, y_test)
y_test = np.sort(y_test)
plt.figure(1, figsize=[50, 50], dpi=20)
plt.plot(x, y_test, color='red', marker='*', label='origin', linestyle='-', linewidth=2)
plt.plot(x, y_return, color='blue', linestyle=':', marker='o',
         label='return', linewidth=1)
plt.show()
# %%
