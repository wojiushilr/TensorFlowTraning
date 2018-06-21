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
#train_data_dir = '/Users/rivaille/Desktop/dataset_new/feng_exp/train/dataset5'
#validation_data_dir= '/Users/rivaille/Desktop/dataset_new/feng_exp/test/dataset1'

train_data_dir = 'C:/Users/USER/Desktop/feng_exp/train/dataset5'
validation_data_dir= 'C:/Users/USER/Desktop/feng_exp/test/dataset1'
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

model1 = Sequential()
model1.add(Conv2D(32, (3, 3), input_shape=input_shape)) #3*3
model1.add(Activation('relu'))
model1.add(Conv2D(32, (3, 3)))
model1.add(Activation('relu'))
model1.add(Conv2D(32, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))#2*2

model1.add(Conv2D(64, (3, 3)))
model1.add(Activation('relu'))
model1.add(Conv2D(64, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(64))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(8))
model1.add(Activation("sigmoid"))





def compile_and_train(model, num_epochs):
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs1/', histogram_freq=0, batch_size=32)
    history = model1.fit_generator(train_generator,
                                   steps_per_epoch=nb_train_samples // batch_size,
                                   epochs=num_epochs,
                                   validation_data=None,
                                   #validation_steps=nb_validation_samples // batch_size,
                                   callbacks=[checkpoint, tensor_board])
    return history

def evaluate_error(model):
    pred = model.predict_generator(validation_generator)
    #pred = np.argmax(pred, axis=1)
    #pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    print('pred',pred)
    print(type(pred))
    print(pred.shape)
    y_pre = to_categorical(pred, num_classes=10)
    print(y_pre.shape)

_ = compile_and_train(model1, num_epochs=20)
evaluate_error(model1)

scoreSeg = model1.evaluate_generator(validation_generator,nb_train_samples // batch_size//2)
print("Accuracy = ",scoreSeg[1])


#model1.save_weights('/Users/rivaille/PycharmProjects/TensorFlowTraning/NN_keras/keras_single_hokousya/weights/model1.h5')

