#edit by LR 20180110
import tensorflow as tf
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average,\
    Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import cifar10
import numpy as np
import os


import numpy as np

##tensorboard --logdir=D:\Lab_Program\TensorFlowTraning\NN_keras\keras_single_hokousya
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

gpu_options = tf.GPUOptions(allow_growth=True)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
y_train = to_categorical(y_train, num_classes=10)

#The dataset consists of 60000 32x32 RGB images from 10 classes. 50000
# images are used for training/validation and the other 10000 for testing
print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(
    x_train.shape, y_train.shape,x_test.shape, y_test.shape))
print(x_test)

input_shape_train = x_train[0,:,:,:].shape
input_shape_label = y_test[0:10,:]
print ("input_shape", input_shape_train)
print ("label", input_shape_label)

model_input = Input(shape=input_shape_train)
print(model_input)


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

        x = Conv2D(64, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block1_conv1')(model_input)
        x = Conv2D(64, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=2)(x)

        # ///////////////////////////////
        x = Conv2D(128, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block2_conv1')(x)
        x = Conv2D(128, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=2)(x)

        # ///////////////////////////////
        x = Conv2D(256, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block3_conv1')(x)
        x = Conv2D(256, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block3_conv2')(x)
        x = Conv2D(256, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=2)(x)

        # ///////////////////////////////
        x = Conv2D(512, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block4_conv1')(x)
        x = Conv2D(512, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block4_conv2')(x)
        x = Conv2D(512, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=2)(x)
        # ///////////////////////////////
        x = Conv2D(512, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block5_conv1')(x)
        x = Conv2D(512, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block5_conv2')(x)
        x = Conv2D(512, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block5_conv3')(x)

        # ///////////////////////////////
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(10)(x)

        x = Activation(activation='softmax')(x)
        model = Model(model_input, x, name='vgg16_1')
        return model

conv_pool_cnn_model = conv_pool_cnn(model_input)
_ = compile_and_train(conv_pool_cnn_model, num_epochs=20)
evaluate_error(conv_pool_cnn_model)