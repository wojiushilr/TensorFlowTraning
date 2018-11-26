#BY LR 20180621
# import the necessary packages
from __future__ import print_function
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import sys
import os
from keras.models import load_model

sys.path.append('..')

#load data/labels from folder with my own rules
def load_data(path):
    print("loading experiment dataset...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    print('imagePaths',imagePaths)
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (96, 64))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = int(imagePath.split(os.path.sep)[-2])
        #print('label',label)
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    #labels = to_categorical(labels, num_classes=8)
    return data,labels



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

#parameter setting
img_width, img_height = 64, 96
epochs = 20
batch_size = 32
train_dir = 'C:\\Users\\USER\Desktop\\data_2\\model1\\train\\'
test_dir = 'C:\\Users\\USER\Desktop\\data_2\\model1\\test\\'

X_train,y_train = load_data(train_dir)
X_test,y_test = load_data(test_dir)

print(X_train.shape)


#data reform to 2D
data = np.append(X_train,X_test)
print(data.shape)
data = np.reshape(data,(16000,-1))
label = np.append(y_train,y_test)

#shuffle the data
#np.random.seed(1024)
index_test = [i for i in range(len(data))]
print("index_test",index_test[0:10])
#np.random.shuffle(index_test)
data = data[index_test]

print(data.shape)
print(label.shape)
print(label[780:790])

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25,random_state=0)

#start training
svc(x_train, y_train, x_test, y_test)
ada(x_train, y_train, x_test, y_test)
rf(x_train, y_train, x_test, y_test)