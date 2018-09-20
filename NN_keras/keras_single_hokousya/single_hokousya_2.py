#BY LR 20180621
# import the necessary packages
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import Model, Input
from imutils import paths
import numpy as np
import random
import cv2
import sys
import os
sys.path.append('..')

#load data/labels from folder with my own rules
def load_data(path):
    print("loading experiment dataset...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    #print('imagePaths',imagePaths)
    #random.seed(30)
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


#parameter setting
img_width, img_height = 64, 96
epochs = 20
batch_size = 32
train_dir2 = 'C:\\Users\\USER\\Desktop\\experiment_data\\model2\\train'
#test_dir2 = 'C:\\Users\\USER\\Desktop\\experiment_data\\model2\\test'
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#data_reading
X_train,y_train = load_data(train_dir2)
#X_test,y_test = load_data(test_dir2)
#y_test = np.argmax(y_test , axis=1)
model_input = Input(shape=input_shape)



#model_2
def model_create(model_input):
    x = Conv2D(32, (3, 3), activation='relu')(model_input)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(8)(x)
    x = Activation(activation='softmax')(x)
    model = Model(model_input, x, name='nin_cnn2')
    return model

def compile_and_train(model, num_epochs):

    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])
    filepath = 'weights2/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs2/', histogram_freq=0, batch_size=batch_size)
    history = model.fit(x=X_train, y=y_train, batch_size=batch_size,
                        epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board],validation_split=0.2)
    return history


model2 = model_create(model_input)
_ = compile_and_train(model2, num_epochs=epochs)
#loss,acc = model2.evaluate(X_test,y_test)
#print('loss,acccccc',loss,acc)

model2.save('model2.h5')