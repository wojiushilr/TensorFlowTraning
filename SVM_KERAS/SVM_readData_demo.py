import read_data as rd
from sklearn.svm import SVC
#import theano
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
'''
def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    #svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    #svcClf.fit(traindata,trainlabel)
    bdt_real = AdaBoostClassifier(
     DecisionTreeClassifier(max_depth=2),
     n_estimators=500,
     learning_rate=0.5)
    bdt_real.fit(traindata, trainlabel)
    pred_testlabel = bdt_real.predict(testdata)
    #pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("svm Accuracy:",accuracy)
    #print(testlabel[100:110])
    #print(pred_testlabel[100:110])
'''
#train_data reading
x_train_0,y_train_0,x_train_1, y_train_1  =  rd.load_train_data()
x_train = np.append(x_train_0,x_train_1)
x_train = np.reshape(x_train,(1258,-1))
y_train = np.append(y_train_0,y_train_1)
y_train = np.reshape(y_train,(-1))
#print(y_train[770:790]) #验证矩阵是否合并成功，合并成功会显示0和1的交界
#test_data reading
x_test_0,y_test_0,x_test_1, y_test_1  =  rd.load_test_data()
x_test = np.append(x_test_0,x_test_1)
x_test = np.reshape(x_test,(320,-1))
y_test = np.append(y_test_0,y_test_1)
y_test = np.reshape(y_test,(-1))
#print(x_test.shape,y_test.shape)
#print(x_train.shape)

# shuffle data
np.random.seed(1024)
index_train = [i for i in range(len(x_train))]
np.random.shuffle(index_train)
data = x_train[index_train]
label = y_train[index_train]

x_train = data[0:1000]
y_train = label[0:1000]
x_test = data[1000:]
y_test = label[1000:]

index_test = [i for i in range(len(x_test))]
np.random.shuffle(index_test)
x_test = x_test[index_test]
y_test = y_test[index_test]


print(x_train.shape)
print(y_test[100:110])

bdt_real = AdaBoostClassifier(
     DecisionTreeClassifier(max_depth=2),
     n_estimators=500,
     learning_rate=0.5)
bdt_real.fit(x_train, y_train)
pred_testlabel = bdt_real.predict(x_test)
#pred_testlabel = svcClf.predict(testdata)
num = len(pred_testlabel)
accuracy = len([1 for i in range(num) if y_test[i]==pred_testlabel[i]])/float(num)
print("svm Accuracy:",accuracy)