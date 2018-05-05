'''
    以相似度作为特征，利用线性回归模型进行预测
    X:三列，分别为横，撇，捺的相似度
    score：四列，分别为横，撇，捺，整字的评分
'''

import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import datasets, linear_model
from sklearn import preprocessing

# generate score：每人写15遍，列为横，撇，捺，大 的评分,最后一行为前面三行的均值±1
# qjy 85-90
# ljb 80-85
# qqq 75-80
# fl  70-75
# lyh zdh 60-70

y = np.zeros([90, 4])
stu = np.zeros(90)

stu[0:15] = 0           # 秦靖尧
stu[15:30] = 1          # 刘佳博
stu[30:45] = 2          # 秦琪琪
stu[45:60] = 3          # 傅良
stu[60:75] = 4          # 李映辉
stu[75:90] = 5          # 朱丹鹤

score_range = [
        [85, 90],
        [80, 85],
        [75, 80],
        ]
# read data
X = np.loadtxt(open('similar_feature_data.csv'), delimiter=',', skiprows=0)
# y = np.loadtxt(open('true_score.csv'), delimiter=',', skiprows=0)
# 标准化
X = preprocessing.scale(X)
# 缩放[0, 1]
min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(X)

for col in range(4):
    R_max=0
    for iter in range(100):
        # 对第col列做线性回归
        # col = 3
        if col <3:          
            # 生成前三列的真实值
            for k in range(3):
                for i in range(k*30,(k+1)*30):
                        y[i,col] = random.randint(score_range[k][0], score_range[k][1])            
        else:               
            # 生成第四列（整字）的真实值
            for k in range(3):
                for i in range(k*30,(k+1)*30):
                    y[i,3] = int(np.mean(y[i,:3]))+random.randint(-1,1)
        
        # 打乱顺序
        index = [i for i in range(len(X))]
        random.shuffle(index)
        X_s = X[index]
        y_s = y[index]
        stu_s = stu[index]       
        
        test_num = 45
        
        # data split
        x_train = X_s[:-test_num]
        y_train = y_s[:-test_num]
        x_test = X_s[-test_num:]
        y_test = y_s[-test_num:]
        stu_test = stu_s[-test_num:]
        
        title = ['Horizontal', 'Left-falling', 'Right-falling', 'Character']
        # 创建模型
        linreg = linear_model.LinearRegression()
        if col <3:
            # 训练 reshape(-1,1)将列向量转换为n*1的矩阵
            linreg.fit(x_train[:,col].reshape(-1,1), y_train[:,col])
            # 预测
            y_predict = linreg.predict(x_test[:,col].reshape(-1,1))
            # 打分
            R = linreg.score(x_test[:,col].reshape(-1,1), y_test[:,col])
        else:
            linreg.fit(x_train, y_train[:,col])
            y_predict = linreg.predict(x_test)
            R = linreg.score(x_test, y_test[:,col])
        if (R>R_max):
            R_max = R
            np.savetxt(title[col]+'_predict.csv', y_predict, delimiter=',')
        
    x_axes = np.arange(y_test.shape[0])
    plt.title(title[col])
    plt.xlabel('sample')
    plt.ylabel('score')
    plt.plot(x_axes, y_test[:,col], color='k',label = 'truth score')
    plt.plot(x_axes, y_predict, color='b', linewidth=3, label = 'predicted score')
    plt.legend(loc='upper right')
    
    plt.show()
    
    # 统计每个人每个笔画的平均真实分值和平均预测分值
    result = np.zeros([6,2])                                    # 每行代表一个人，第一列代表真实值，第二列代表预测值
    for k in range(6):
        # 对第k个人进行统计
        people_index = np.where(stu_test == k)[0]               # 索引     
        people_count = len(people_index)                        # 样本个数
        people_test = np.transpose(y_test[people_index,col])      # 真实分值列表
        people_predict = y_predict[people_index]                # 预测分值列表
        result[k, 0] = sum(people_test)//people_count           # 真实平均值
        result[k, 1] = sum(people_predict)//people_count        # 预测平均值
    print(result)
    print('当前的R值为', R_max)
np.savetxt('y_test.csv',y_test, delimiter=',')

# --------------- 独立的绘图 ------
    
#import matplotlib.pyplot as plt
#import numpy as np
#title = ['heng', 'pie', 'na', 'whole']
#col =  0
#x_axes = np.arange(y_test.shape[0])
#plt.title(title[col])
#plt.xlabel('sample')
#plt.ylabel('score')
#plt.plot(x_axes, y_test[:,col], color='k',label = 'true')
#plt.plot(x_axes, heng_predictcsv, color='b', linewidth=3, label = 'predict')
#plt.legend(loc='upper right')    
