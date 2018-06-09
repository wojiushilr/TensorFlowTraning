#edit by LR 20180110
#BUG exit
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
#86个特征量 20类
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
x_train = x_train[:,:,np.newaxis,np.newaxis]
x_test = x_test[:,:,np.newaxis,np.newaxis]
print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(
    x_train.shape, y_train.shape,x_test.shape, y_test.shape))

input_shape_train = x_train[0,:,:,:].shape
print(input_shape_train)
model_input = Input(shape=input_shape_train)



model = Sequential(name='simple_MLP')

#First model: ConvPool-CNN-C
def classic_neural(model_input):
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)

    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='conv_pool_cnn')

    return model

classic_neural_model = classic_neural(model_input)



def compile_and_train(model, num_epochs):
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=32)
    history = model.fit(x=x_train, y=y_train, batch_size=32,
                        epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    return history
_ = compile_and_train(classic_neural_model, num_epochs=1)

def evaluate_error(model):
    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
    return error
print (evaluate_error(classic_neural_model))



