# %%
import re


def ReadTxt(FileName):
    EMGList = [[], [], [], [], [], [], [], []]
    with open(FileName, 'r') as f:
        lines = f.readlines()
        for line in lines:
            [name, data] = line.split("\t", 1)
            if name == "RawEMG:":
                EMG = [int(s) for s in re.findall(r'\d+', data)]
                for i in range(0, 8):
                    EMGList[i].append(EMG[i])
    return EMGList


# %%
if __name__ == '__main__':
    filename = "E:/txt/Desktop/file/task/20210803 main/data/静息.txt"
    print(ReadTxt(filename))
# %%
