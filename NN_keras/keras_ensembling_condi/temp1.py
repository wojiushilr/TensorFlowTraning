#edit by LR 20180110
#available to run
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import mnist
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications import imagenet_utils
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense

data= pd.read_csv('feature_values.csv')
X = data[['BL1','BL2','BL3','BL4','BL5','BL6','BR1','BR2','BR3','BR4','BR5','BR6',
          'EL1','EL2','EL3','EL4','EL5','EL6','ER1','ER2','ER3','ER4','ER5','ER6',
          'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','M1','M2','M3','M4',
          'M5','M6','M7','M8','M9','N2','N3','N4','N5','MFCC1','MFCC2','MFCC3',
          'MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9','MFCC10','MFCC11',
          'MFCC12','E','Del1','Del2','Del3','Del4','Del5','Del6','Del7','Del8',
          'Del9','Del10','Del11','Del12','DelE','Acc1','Acc2','Acc3','Acc4',
          'Acc5','Acc6','Acc7','Acc8','Acc9','Acc10','Acc11','Acc12','AccE']]
y = data[['NAME']]


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)



print(type(x_test))
#x_train = x_train[:,:,np.newaxis,np.newaxis]
#x_test = x_test[:,:,np.newaxis,np.newaxis]
print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(
    x_train.shape, y_train.shape,x_test.shape, y_test.shape))


batch_size = 20  # mini_batch_size
nb_epoch = 500  # 大循环次数

model_shape = x_train[0,:].shape
print(model_shape)

print('Building simple_MLP model...')
model = Sequential(name='simple_MLP') # 第一层<br>#Dense就是全连接层
model.add(Dense(40, input_shape=(model_shape)))  # 输入维度, 512==输出维度
model.add(Activation('relu'))  # 激活函数
model.add(Dropout(0.5))  # dropout<br><br>#第二层
model.add(Dense(20))
model.add(Activation('softmax'))
# 损失函数设置、优化函数，衡量标准



def compile_and_train(model, num_epochs):

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer='adam',
                  metrics=['accuracy'])

    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=20)
    # 训练，交叉验证
    history = model.fit(x_train, y_train,
                        epochs=num_epochs, batch_size=20, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    score = model.evaluate(x_test, y_test,
                           batch_size=20, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return history
_ = compile_and_train(model, num_epochs=nb_epoch)


def evaluate_error(model):
    pred = model.predict(x_test, batch_size = 20)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
    return error

print ('Test error:',evaluate_error(model))





