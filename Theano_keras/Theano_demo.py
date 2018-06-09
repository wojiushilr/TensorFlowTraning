from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers import BatchNormalization
from sklearn.svm import SVC
#import theano
from keras.utils import np_utils
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')

def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)

    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-svm Accuracy:",accuracy)

#each add as one layer
model = Sequential()

#1 .use convolution,pooling,full connection
model.add(Conv2D(5, (3, 3), activation="tanh", padding="valid", input_shape=(1, 28, 28)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(10, (3, 3), activation="tanh"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(100,activation='tanh')) #Full connection

model.add(Dense(10,activation='softmax'))

#2 .just only user full connection
# model.add(Dense(100,input_dim = 784, init='uniform',activation='tanh'))
# model.add(Dense(100,init='uniform',activation='tanh'))
# model.add(Dense(10,init='uniform',activation='softmax'))

# sgd = SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy' ,metrics=['accuracy'])

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train.shape",X_train.shape)
print("X_test.shape",X_test.shape)
print("y-train.shape",y_train.shape)
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1],X_train.shape[2]) #注意这种用法

Y_train = np_utils.to_categorical(y_train, 10)
# keras自带工具，keras.utils. np_utils可以完成转换，例如，若y_test为整型的类别标签，
# Y_test = np_utils.to_categorical(y_test, nb_classes)， Y_test将得到0,1序列化的结果。

print("Y_train.shape",Y_train.shape)

#new label for svm
y_train_new = y_train[0:42000]
y_test_new = y_train[42000:]

#new train and test data
X_train_new = X_train[0:42000]
X_test = X_train[42000:]
Y_train_new = Y_train[0:42000]
Y_test = Y_train[42000:]
print(len(X_train))
print(Y_train_new.shape)
'''
model.fit(X_train_new, Y_train_new, batch_size=200, epochs=1,shuffle=True, verbose=1, validation_split=0.2)
print("Validation...")
val_loss,val_accuracy = model.evaluate(X_test, Y_test, batch_size=1)
print ("val_loss: %f" %val_loss)
print ("val_accuracy: %f" %val_accuracy)
'''
#get output of FC layer
get_feature = K.function([model.layers[0].input,K.learning_phase()],[model.layers[5].output])
FC_train_feature0 = get_feature([X_train_new])
FC_test_feature0 = get_feature([X_test])
print(type(FC_test_feature0))
print(np.array(FC_test_feature0).shape)
print(np.array(FC_train_feature0).shape)
#print(FC_train_feature0)
x,y,z = np.array(FC_train_feature0).shape
n,m,p = np.array(FC_test_feature0).shape
FC_train_feature = np.array(FC_train_feature0).reshape(x*y,z)
FC_test_feature = np.array(FC_test_feature0).reshape(n*m,p)

print(FC_train_feature.shape)
'''
FC_train_feature = np.reshape(FC_train_feature0,(-1,len(FC_train_feature0)))
FC_test_feature = np.reshape(FC_test_feature0,(-1,len(FC_train_feature0)))
print(FC_test_feature)'''
print(model.layers[6].output)
print(get_feature)
svc(FC_train_feature,y_train_new,FC_test_feature,y_test_new)