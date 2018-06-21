#edit by LR 20180110
import tensorflow as tf
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import cifar10
import numpy as np
import os
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential


import numpy as np

##tensorboard --logdir=/Users/rivaille/PycharmProjects/TensorFlowTraning/NN_keras/keras_ensemblng_pic/logs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

gpu_options = tf.GPUOptions(allow_growth=True)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


###
img_width, img_height = 64, 96
batch_size=32
train_data_dir = '/Users/rivaille/Desktop/dataset_new/feng_exp/train/dataset5'
validation_data_dir= '/Users/rivaille/Desktop/dataset_new/feng_exp/test/dataset1'
nb_train_samples = 1280
nb_validation_samples = 320
#data_format,tensorflow of theaon
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

'''
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
y_train = to_categorical(y_train, num_classes=10)
'''


#The dataset consists of 60000 32x32 RGB images from 10 classes. 50000
# images are used for training/validation and the other 10000 for testing
print(input_shape)
print(type(train_data_dir))


'''

input_shape_train = x_train[0,:,:,:].shape
input_shape_label = y_test[0:10,:]
print ("input_shape", input_shape_train)
print ("label", input_shape_label)
'''

train_datagen = ImageDataGenerator(rescale=1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

print("validation_generator.class_indices:",type(validation_generator.class_indices),validation_generator.class_indices)

#First model: ConvPool-CNN-C
def conv_pool_cnn():
    input_img = Input(shape=(64,96,3))
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    #x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    #x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    #x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    #x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(8, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    model = Model(input_img, x, name='all_cnn')
    return model


def compile_and_train(model, num_epochs):
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs1/', histogram_freq=0, batch_size=32)
    history = model.fit_generator(train_generator,
                                   steps_per_epoch=nb_train_samples // batch_size,
                                   epochs=num_epochs,
                                   validation_data=validation_generator,
                                   validation_steps=nb_validation_samples // batch_size,
                                   callbacks=[checkpoint, tensor_board])
    return history

def evaluate_error(model):
    pred = model.predict_generator(validation_generator)
    #pred = np.argmax(pred, axis=1)
    #pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    print('pred',pred)
    print(type(pred))
    print(pred.shape)
    print()

#input_shape_train = x_train[0,:,:,:].shape
conv_pool_cnn_model = conv_pool_cnn()
_ = compile_and_train(conv_pool_cnn_model, num_epochs=1)
evaluate_error(conv_pool_cnn_model)


conv_pool_cnn_model.save_weights('/Users/rivaille/PycharmProjects/TensorFlowTraning/NN_keras/keras_single_hokousya/weights/model1.h5')