'''
20181110
LIRUI
'''

#必要なライブラリをセットアップ
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adadelta

#Mnistデータ（訓練データ）を読み込み
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#データをリサイズする
X_train = X_train.reshape(X_train.shape[0],28,28,1).astype("float32")
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype("float32")
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#
num_class = 10
batch_size = 32

#モデル構築　Start
model = Sequential()
#
model.add(Conv2D(filters = 64, kernel_size= (3,3), activation="relu",input_shape= (28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
#
model.add(Conv2D(filters = 64, kernel_size= (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
#
model.add(Flatten())
#
model.add(Dense(128, activation="relu"))
model.add(Dense(num_class, activation="softmax"))
#
model.compile(loss= "categorical_crossentropy", optimizer=Adadelta(), metrics=["accuracy"])

#
model.fit(X_train, y_train, batch_size=batch_size,
          verbose = 1, validation_data=(X_test, y_test))

#
score = model.evaluate(X_test, y_test, verbose= 0)
print("test loss", score[0])
print("test accuracy", score[1])

#
model.save("mnist_demo.h5")