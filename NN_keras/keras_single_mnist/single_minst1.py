#edit by LR 20180110
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.applications import imagenet_utils
import six

import numpy as np
data = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
y_train = y_train [:20000]
y_test = y_test[:2000]
x_train = x_train[:20000]
x_test = x_test[:2000]

y_train = to_categorical(y_train, num_classes=10)

#reshape mnist data
x_train = x_train[:,:,:,np.newaxis]
x_test = x_test[:,:,:,np.newaxis]
y_test = y_test[:,np.newaxis]
#The dataset consists of 60000 32x32 RGB images from 10 classes. 50000
# images are used for training/validation and the other 10000 for testing
print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(
    x_train.shape, y_train.shape,x_test.shape, y_test.shape))
print(type(x_test))

input_shape_train = x_train[0,:,:,:].shape
input_shape_label = y_test[0:100,:]
print ("input_shape", input_shape_train)
print ("label", input_shape_label)

model_input = Input(shape=input_shape_train)


def compile_and_train(model, num_epochs):
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs1/', histogram_freq=0, batch_size=32)
    history = model.fit(x=x_train, y=y_train, batch_size=32,
                        epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    return history

def evaluate_error(model):
    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
    return error

#First model: ConvPool-CNN-C
def conv_pool_cnn(model_input):
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    model = Model(model_input, x, name='conv_pool_cnn')
    return model
conv_pool_cnn_model = conv_pool_cnn(model_input)
_ = compile_and_train(conv_pool_cnn_model, num_epochs=5)
evaluate_error(conv_pool_cnn_model)