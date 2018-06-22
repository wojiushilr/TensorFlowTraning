#BY LR 20180621
# import the necessary packages
import matplotlib
matplotlib.use("Agg")
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
import numpy as np
import os
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import os
import sys
sys.path.append('..')

#load data/labels from folder with my own rules
def load_data(path):
    print("loading experiment dataset2...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    #print('imagePaths',imagePaths)
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
        print('label',label)
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=8)
    return data,labels

X_train,y_train = load_data('/Users/rivaille/Desktop/dataset_new/feng_exp/train')
X_test,y_test = load_data('/Users/rivaille/Desktop/dataset_new/feng_exp/test')
y_test = np.argmax(y_test , axis=1)

print(X_train.shape)

#parameter setting
img_width, img_height = 64, 96
epochs = 20
batch_size = 32
#data_format,tensorflow of theaon
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#model_2
def model_create(shape):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=shape)) #3*3
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))#2*2

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8))
    model.add(Activation("softmax"))
    return model

def compile_and_train(model, num_epochs):

    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs1/', histogram_freq=0, batch_size=batch_size)
    history = model.fit(x=X_train, y=y_train, batch_size=batch_size,
                        epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    return history

def evaluate_error(model):

    pred = model.predict(X_test, batch_size = batch_size)
    pred = np.argmax(pred, axis=1)
    print(pred.shape)
    #pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    #pred = to_categorical(pred, num_classes=10)
    print(y_test.shape[0])
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
    return error


model2 = model_create(input_shape)
_ = compile_and_train(model2, num_epochs=epochs)
err=evaluate_error(model2)
print('error',err)