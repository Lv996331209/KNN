from sklearn.datasets.base import load_wine as dt
import pandas as pd



wineDataSet=dt()


df=pd.DataFrame(wineDataSet['data']
                ,columns=['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'])

print(wineDataSet['data'])
print("红酒数据集中的键：\n{}".format(wineDataSet.keys()))
list=[]
for i in (wineDataSet['target']):
    if i==0:
        list.append(1)
    elif i==1:
        list.append(2)
    elif i==2:
        list.append(3)
print(len(list))
df['Class']=list
#print(df)
#df.to_csv('wine.csv')


#print("红酒数据集中的键：\n{}".format(wineDataSet.keys()))
#print("数据概况：\n{}".format(wineDataSet['data'].shape))