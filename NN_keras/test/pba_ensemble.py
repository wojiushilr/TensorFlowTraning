

from keras.utils.np_utils import to_categorical
import numpy as np
import os
from PIL import Image

import keras
from keras.models import Model, Input
from keras.layers import Average
from keras.models import load_model

from keras import backend as K
#parameter setting
img_width, img_height = 64, 96 #
nb_train_samples = 6400
nb_test_samples = 1600

train_data_dir = 'C:\\Users\\USER\\Desktop\\dataset1\\train\\'
test_data_dir = 'C:\\Users\\USER\\Desktop\\dataset1\\test\\'

def load_data( nb_samples, height, width, chanel, path):
    data = np.empty((nb_samples, height, width, chanel),dtype="float32")
    label = np.empty((nb_samples,),dtype="uint8")

    imgs = os.listdir(path)
    num = len(imgs)
    for i in range(num):
        img = Image.open(path+imgs[i])
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        label[i] = int(imgs[i].split('.')[0])
    return data,label

# *** read ****
x_train, y_train = load_data(nb_train_samples, img_height, img_width, 3,train_data_dir)
x_train = x_train.reshape(x_train.shape[0], 96, 64, 3).astype('float32') / 255
#print(x_train.shape[0], ' trainsamples')
x_test, y_test = load_data(nb_test_samples, img_height, img_width, 3,test_data_dir)
x_test = x_test.reshape(x_test.shape[0], 96, 64, 3).astype('float32') / 255
#print(x_test.shape[0], ' testsamples')

print(x_train.shape)

##############################


input_shape = (img_height, img_width , 3)

input_shape_train = input_shape
print(input_shape_train)

def evaluate_error(model):
    score = model.evaluate(x_test, y_test)
    return score

def evaluate_error(model):
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    #pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
    return error

model_input = Input(shape=input_shape_train)


def ensemble(models, model_input):

    outputs = [model(model_input) for model in models]
    #outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    model = Model(model_input, y, name='ensemble')
    return model



#ensemble model

model1 = load_model('model1.h5')
model2 = load_model('model2.h5')
model3 = load_model('model3.h5')
model4 = load_model('model4.h5')
model5 = load_model('model5.h5')
models=[model4,model5]


ensemble_model = ensemble(models, model_input)


result = 1- evaluate_error(ensemble_model)

print(result)