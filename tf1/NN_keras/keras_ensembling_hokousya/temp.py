#BY LR 20180621
# import the necessary packages
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model, Input
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import sys
import os
sys.path.append('..')


#load data/labels from folder with my own rules
def load_data(path):
    print("loading experiment dataset1...")
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
        #print('label',label)
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
train_dir = 'C:\\Users\\USER\\Desktop\\experiment_data\\model2\\train'
test_dir = 'C:\\Users\\USER\\Desktop\\experiment_data\\model2\\test'
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#data_reading
X_train,y_train = load_data(train_dir)
X_test,y_test = load_data(test_dir)
y_test = np.argmax(y_test , axis=1)
print('y_test',y_test)

print(X_train.shape)


#ensemble model

model1 = load_model('model2.h5')
models=[model1,model1]

def ensemble(models, model_input):

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input , y , name="ensemble")
    return model

model_input = Input(shape=input_shape)
ensemble_model = ensemble(models,model_input)

pred = ensemble_model.predict(X_test)
pred = np.argmax(pred, axis=1)
print(pred)
error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
print(error)