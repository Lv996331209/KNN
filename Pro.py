import pandas as pd
import numpy as np
import operator  # 运算符模块
from numpy import *
import datetime

#数据读取与预处理
def preprocessing():
    df=pd.read_csv('wine.csv')#读入原始数据
    df=df.sample(frac = 1) #打乱原始数据
    #按照7：3比例划分样本数据与测试数据
    train_len=int(0.7*len(df))
    train=df[:train_len]#样本数据124条
    test=df[train_len:]#测试数据54条
    #样本数据与测试数据分别存储
    train.to_csv('train.csv')
    test.to_csv('test.csv')

# 归一化特征值 公式：(当前值-最小值)/range
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

#Dataframe格式转换np.array
def df2array(train):
    train = train.drop('Class', axis=1)#删去分类信息，只处理数据
    train = np.array(train)#格式转换为up.array
    Mat = np.delete(train, [0, 1], axis=1)#删去index
    return Mat

#统计近邻列表
def classify(list):
    s = set(list)#列表去重
    dict = {}#建立字典
    #循环统计各个元素重复个数
    for item in s:
        dict.update({item: list.count(item)})
    print(dict)
    #字典排序查找最近邻
    sortedClassCount = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)  # 排序频率
    #返回最近邻分类
    return sortedClassCount[0][0]

#距离度量 DataMat为样本数据,inArr为待分类数据
def distance(DataMat,inArr,train_class,K):
    #样本数据归一化
    normMat, ranges, minVals = autoNorm(DataMat)
    #计算欧氏距离
    Mat = (sqrt(((inArr - minVals) / ranges - normMat) ** 2))
    result = Mat.sum(axis=1)
    #降序排列提取索引值
    index= result.argsort()
    # 选取距离最近的前K个值
    index= list(index[:K])
    #按照索引值建立K近邻列表
    neighbour=[]
    for i in index:
        neighbour.append(train_class[int(i)])
    #返回K近邻列表
    return neighbour

#KNN分类测试 DataMat为样本数据,TestMat为测试数据
def ClassTest(DataMat,TestMat,test,train_class,K):
    test_class = []#建立分类表
    #测试数据依次分类
    for inArr in TestMat:
        neighbour = distance(DataMat, inArr,train_class,K)
        target = classify(neighbour)
        test_class.append(target)
    #分类结果写入测试数据文件
    test['test_Class'] = test_class
    #test.to_csv('test_classify.csv')
    #对比实际分类，计算KNN分类准确率
    true_class=test['Class']
    Count=0
    for item in range(len(true_class)):
        if true_class[item]==test_class[item]:
            Count+=1
    #print("分类正确:"+str(Count)+"条")
    #print("总共分类:"+str(len(true_class))+"条")
    #print("分类准确率:"+str(round(Count*100/len(true_class),2))+"%")

    return Count/len(true_class)
if __name__ == '__main__':
    #读取训练样本数据与测试数据


    starttime = datetime.datetime.now()

    train=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')
    #提取训练样本分类列表
    train_class=list(train['Class'])
    #训练样本数据与测试数据转化为np.array
    DataMat=df2array(train)
    TestMat=df2array(test)
    #执行分类测试
    ClassTest(DataMat,TestMat,test,train_class,K=13)
    endtime = datetime.datetime.now()

    print(endtime - starttime).seconds






