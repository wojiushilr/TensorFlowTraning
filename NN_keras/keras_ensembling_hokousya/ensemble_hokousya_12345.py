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
epochs = 10
batch_size = 32
train_dir = 'C:\\Users\\USER\Desktop\\data_2\\model1\\train\\'
test_dir = 'C:\\Users\\USER\Desktop\\data_2\\model1\\test\\'
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

print(input_shape)

#data_reading
X_train,y_train = load_data(train_dir)

X_test,y_test0 = load_data(test_dir)
y_test = np.argmax(y_test0 , axis=1)
model_input = Input(shape=input_shape)

print(model_input)


#ensemble model

model1 = load_model('model2.h5')
model2 = load_model('model22.h5')
model3 = load_model('model3.h5')
model4 = load_model('model4.h5')
model5 = load_model('model5.h5')
models=[model1,model2,model3,model4,model5]

def ensemble(models, model_input):

    outputs = [model(model_input) for model in models]

    #outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input , y , name="ensemble")
    return model

def compile_and_train(model, num_epochs):

    #model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
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


ensemble_model = ensemble(models,model_input)
#_ = compile_and_train(ensemble_model,num_epochs=epochs)
err = evaluate_error(ensemble_model)
print('error',err)
print('acc',1-err)

loss1,acc1 = model1.evaluate(X_test,y_test0)
loss2,acc2 = model2.evaluate(X_test,y_test0)
loss3,acc3 = model3.evaluate(X_test,y_test0)
loss4,acc4 = model4.evaluate(X_test,y_test0)
loss5,acc5 = model5.evaluate(X_test,y_test0)

print('loss1,acccccc1',loss1,acc1)
print('loss2,acccccc2',loss2,acc2)
print('loss3,acccccc3',loss3,acc3)
print('loss4,acccccc4',loss4,acc4)
print('loss5,acccccc5',loss5,acc5)
print('ensemble error',err)
print('ensemble acc',1-err)
