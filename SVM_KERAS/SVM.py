from keras import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers import BatchNormalization
from sklearn.svm import SVC
from keras.utils import np_utils
import numpy as np
from keras import backend as K


def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)

    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-svm Accuracy:",accuracy)


# dimensions of our images.
img_width, img_height = 64, 128

train_data_dir = 'C:\\Users\\Rivaille\\Desktop\\dataset3\\feng_1\\feng_learn'
validation_data_dir = 'C:\\Users\\Rivaille\\Desktop\\dataset3\\feng_1\\feng_test'
nb_train_samples = 1258
nb_validation_samples = 320
epochs = 1
batch_size = 50

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
#1 .use convolution,pooling,full connection
model.add(Conv2D(32, (3, 3), activation="relu", padding="valid", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64,activation='relu')) #Full connection

model.add(Dense(1,activation='sigmoid'))

#2 .just only user full connection
# model.add(Dense(100,input_dim = 784, init='uniform',activation='tanh'))
# model.add(Dense(100,init='uniform',activation='tanh'))
# model.add(Dense(10,init='uniform',activation='softmax'))

# sgd = SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='rmsprop', loss='binary_crossentropy' ,metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    classes=["0","1"],
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    classes=["0","1"],
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
'''
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
'''
#model.save_weights('Cnn.h5')
model.summary()
print(model.layers[5].output)


get_feature = K.function([model.layers[0].input,K.learning_phase()],[model.layers[5].output])
FC_train_feature0 = get_feature([train_generator])[0]
#FC_test_feature0 = get_feature([validation_generator])
print(get_feature)
print(FC_train_feature0.shape)
'''
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
print(X_train_new.shape)

model.fit(X_train_new, Y_train_new, batch_size=200, epochs=1,shuffle=True, verbose=1, validation_split=0.2)
print("Validation...")
val_loss,val_accuracy = model.evaluate(X_test, Y_test, batch_size=1)
print ("val_loss: %f" %val_loss)
print ("val_accuracy: %f" %val_accuracy)

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

svc(FC_train_feature,y_train_new,FC_test_feature,y_test_new)

'''
