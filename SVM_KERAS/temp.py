from __future__ import print_function

import os
from PIL import Image
import numpy as np
import pickle as p
import matplotlib.pyplot as pyplot
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
'''
def img2vector(imgFile):
    img = Image.open(imgFile).convert('L')
    img_arr = np.array(img, 'i') # 20px * 20px 灰度图像
    img_normlization = np.round(img_arr/255) # 对灰度值进行归一化
    img_arr2 = np.reshape(img_normlization, (1,-1)) # 1 * 400 矩阵
    return img_arr2

'''
num1 = 6400

def image_to_array_0(j):
    """
    图片转化为数组并存为二进制文件；
    :param filenames:文件列表
    :return:
    """
    image_base_path = "/Users/rivaille/Desktop/experiment_data/model1/train/{}/".format(j)
    print(image_base_path)

    filenames_0 = os.listdir("/Users/rivaille/Desktop/experiment_data/model1/train/{}/".format(j))
    n = len(filenames_0) # 获取图片的个数
    result = np.array([])  # 创建一个空的一维数组
    label = np.empty((800,),dtype="int")

    for i in range(n):
        image = Image.open(image_base_path + filenames_0[i])
        r, g, b = image.split()  # rgb通道分离
        # 注意：下面一定要reshpae(8192)使其变为一维数组，否则拼接的数据会出现错误，导致无法恢复图片
        r_arr = np.array(r).reshape(6144)
        g_arr = np.array(g).reshape(6144)
        b_arr = np.array(b).reshape(6144)
        # 行拼接，类似于接火车；最终结果：共n行，一行24576列，为一张图片的rgb值
        image_arr = np.concatenate((r_arr, g_arr, b_arr))
        result = np.concatenate((result, image_arr))
        #print(result)
        label[i] = int(j)
    print("start pic transform to array of label:{label}".format(label=label[0]))
    result = result.reshape((n, 18432))
    print("transform success!")
    #print(result.shape)
    #file_path = self.data_base_path + "data2.bin"
    #with open(file_path, mode='wb') as f:
    #    p.dump(result, f)
    #print("保存文件成功")
    return  result,label
'''
def image_to_array_1():
    """
    图片转化为数组并存为二进制文件；
    :param filenames:文件列表
    :return:
    """
    image_base_path = "C:\\Users\\Rivaille\\Desktop\\dataset3\\feng_1\\feng_learn\\1\\"
    filenames_0 = os.listdir("C:\\Users\\Rivaille\\Desktop\\dataset3\\feng_1\\feng_learn\\1")
    n = len(filenames_0) # 获取图片的个数
    result = np.array([])  # 创建一个空的一维数组
    label = np.empty((480,),dtype="int")

    for i in range(n):
        image = Image.open(image_base_path + filenames_0[i])
        r, g, b = image.split()  # rgb通道分离
        # 注意：下面一定要reshpae(1024)使其变为一维数组，否则拼接的数据会出现错误，导致无法恢复图片
        r_arr = np.array(r).reshape(8192)
        g_arr = np.array(g).reshape(8192)
        b_arr = np.array(b).reshape(8192)
        # 行拼接，类似于接火车；最终结果：共n行，一行3072列，为一张图片的rgb值
        image_arr = np.concatenate((r_arr, g_arr, b_arr))
        result = np.concatenate((result, image_arr))
        #print(result)
        label[i] = int(1)
    print("start pic transform to array of label:{label}".format(label=label[0]))
    result = result.reshape((n, 24576))  # 将一维数组转化为count行3072列的二维数组
    print("transform success!")
    print(result.shape)
    #file_path = self.data_base_path + "data2.bin"
    #with open(file_path, mode='wb') as f:
    #    p.dump(result, f)
    #print("保存文件成功")
    return  result,label
'''

def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=0.5,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)
    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("svm Accuracy:",accuracy)
    print(testlabel[100:110])
    print(pred_testlabel[100:110])

def ada(traindata,trainlabel,testdata,testlabel):
    print("Start training Adaboost...")
    bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=5, learning_rate=0.8)
    bdt_real.fit(traindata, trainlabel)
    pred_testlabel = bdt_real.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("adaboost Accuracy:",accuracy)
    print(testlabel[100:110])
    print(pred_testlabel[100:110])

def rf(traindata,trainlabel,testdata,testlabel):
    randomf = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
                                  min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)
    randomf.fit(traindata, trainlabel)
    pred_testlabel = randomf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("random forest Accuracy:",accuracy)
    print(testlabel[100:110])
    print(pred_testlabel[100:110])

#get data and label
data = []
label = []
for i in range(8):

    x, y = image_to_array_0(i)
    data.append(x)
    label.append(y)

    print(x.shape)
    print(y.shape)

#print(x_0.shape)
#print(x_1.shape)
#print(y_0[100:110])
#print(y_1[100:110])

#data reform to 2D
data = np.reshape(data,(num1,-1))
label = np.reshape(label,(num1))
#shuffle the data
#np.random.seed(1024)
np.random.shuffle(data)
np.random.shuffle(label)


print(data.shape)
print(label[780:790])

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25,random_state=0)
'''
#distribute train and test data
x_train = data[0:800]
y_train = label[0:800]
x_test = data[800:]
y_test = label[800:]'''



'''
bdt_real = AdaBoostClassifier(
     DecisionTreeClassifier(max_depth=2),
     n_estimators=500,
     learning_rate=0.5)
bdt_real.fit(x_train, y_train)
pred_testlabel = bdt_real.predict(x_test)
#pred_testlabel = svcClf.predict(testdata)
num = len(pred_testlabel)
accuracy = len([1 for i in range(num) if y_test[i]==pred_testlabel[i]])/float(num)
print("Adaboost Accuracy:",accuracy)'''

#start training


svc(x_train, y_train, x_test, y_test)
ada(x_train, y_train, x_test, y_test)
rf(x_train, y_train, x_test, y_test)