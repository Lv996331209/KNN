from Pro import df2array,ClassTest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
#提取训练样本分类列表
train_class=list(train['Class'])
#训练样本数据与测试数据转化为np.array
DataMat=df2array(train)
TestMat=df2array(test)
#执行分类测试
rate=[]
x=[]
for i in range(1,21):
    x.append(i)
    rate.append(ClassTest(DataMat,TestMat,test,train_class,K=i))
print(rate)
print(sum(rate)/len(rate))

plt.stem(x,rate,linefmt=':')#做取样图
x_ticks = np.linspace(1,20,20)#限定x轴刻度
y_ticks = np.linspace(0.9,1,11)#限定y轴刻度
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.xlim(0,21,1)#限定x轴范围
plt.ylim(0.9,1,0.01)#限定x轴范围
plt.title('K值与分类准确率关系',fontproperties=font,size=15)#写入标题
plt.xlabel('K值', fontproperties=font,size=12)#写入x轴标签名称
plt.ylabel('分类准确率', fontproperties=font,size=12)#写入y轴标签名称
y=[]
for xy in zip(x, rate):
    plt.annotate(s="(%s)" % round(xy[1],2), xy=xy, xytext=(-15, 10), textcoords='offset points')
#plt.show()