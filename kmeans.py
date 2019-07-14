import numpy as np

import pandas as pd
from sklearn.cluster import KMeans#导入kmeans库
from scipy.io import loadmat
import matplotlib.pyplot as plt
# from skimage import io
import pandas as pd
import numpy as np
import operator  # 运算符模块
from numpy import *
import datetime

def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 存放每列最小值，参数0使得可以从列中选取最小值，而不是当前行
    maxVals = dataSet.max(0)  # 存放每列最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))  # 初始化归一化矩阵为读取的dataSet
    m = dataSet.shape[0]  # m保存第一行
    # 特征矩阵是13x124，min max range是1x3 因此采用tile将变量内容复制成输入矩阵同大小
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
def df2array(train):
    train = train.drop('Class', axis=1)#删去分类信息，只处理数据
    train = np.array(train)#格式转换为up.array
    Mat = np.delete(train, [0, 1], axis=1)#删去index
    return Mat

df=pd.read_csv('wine.csv')
train=df2array(df)
train,_,_=autoNorm(train)
print(train.shape)
model = KMeans(n_clusters=3, n_init=100, n_jobs=-1)
model.fit(train)
print(model.predict(train))