#BY LR 20180621
# import the necessary packages
from __future__ import print_function
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import sys
import os
import matplotlib.pyplot as plt
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


def PlotGridSearchScores(model_tuning, x_param, line_param):
    x_values = model_tuning.cv_results_['param_' + x_param].data
    x_labels = np.sort(np.unique(x_values))
    x_keys = ['{0:9.2e}'.format(x) for x in x_labels]

    line_values = model_tuning.cv_results_['param_' + line_param].data
    line_labels = np.sort(np.unique(line_values))
    line_keys = ['{0:9.2e}'.format(v) for v in line_labels]

    score = {}

    # (line_key, x_key) -> mean_test_scoreを生成
    for i, test_score in enumerate(model_tuning.cv_results_['mean_test_score']):
        x = x_values[i]
        line_value = line_values[i]

        x_key = '{0:9.2e}'.format(x)
        line_key = '{0:9.2e}'.format(line_value)

        score[line_key, x_key] = test_score

    _, ax = plt.subplots(1, 1)

    # 対数軸で表示する
    plt.xscale('log')

    # x_paramをx軸、line_paramを折れ線グラフで表現
    for line_key in line_keys:
        line_score = [score[line_key, x_key] for x_key in x_keys]
        ax.plot(x_labels, line_score, '-o', label=line_param + ': ' + line_key)

    ax.set_title("Grid Search Accuracy Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(x_param, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 0.95, 0.5, .100), fontsize=15)
    ax.grid('on')



#parameter setting
train_dir = 'C:\\Users\\USER\\Desktop\\data_2\\model2\\train'
test_dir = 'C:\\Users\\USER\\Desktop\\data_2\\model2\\test'

X_train,y_train = load_data(train_dir)
X_test,y_test = load_data(test_dir)

print(X_train.shape)


#data reform to 2D
data = np.append(X_train,X_test)
print(data.shape)
data = np.reshape(data,(8000,-1))
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

print("Start training SVM...")
'''
estimator = SVC()
classifier = OneVsRestClassifier(estimator= estimator)
#svcClf = SVC(C=0.5,kernel="rbf",cache_size=3000)
classifier.fit(x_train,y_train)
pred_testlabel = classifier.predict(x_test)
num = len(pred_testlabel)
#accuracy = len([1 for i in range(num) if y_test[i]==pred_testlabel[i]])/float(num)
#print("svm Accuracy:",accuracy)
print(y_test[100:110])
print(pred_testlabel[100:110])
'''

model = RandomForestClassifier()

C_params = [1,10,100]
#gamma_params = [0.01,0.001,0.0001,0.00001]
max_depth = [13,14,15,16,17,18]
parameters = {
    'n_estimators': C_params,
    'max_depth': max_depth

}

model_tuning = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    verbose=3
)

model_tuning.fit(x_train, y_train)


# チューニング結果を描画
PlotGridSearchScores(model_tuning, 'n_estimators', 'max_depth')


# Best parameter
print('best:',model_tuning.best_params_)
# 評価データでconfusion matrixとaccuracy scoreを算出
classifier_tuned = model_tuning.best_estimator_
pred = classifier_tuned.predict(x_test)

accuracy = accuracy_score(y_test, pred)
print('Multiclass (default): %.3f' % accuracy)
plt.show()