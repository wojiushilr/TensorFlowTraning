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

y_train = y_train [:50000]
y_test = y_test[:10000]
x_train = x_train[:50000]
x_test = x_test[:10000]

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
    filepath = 'weights2/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs4/', histogram_freq=0, batch_size=32)
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
#Second model: ALL-CNN-C
def all_cnn(model_input):
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    model = Model(model_input, x, name='all_cnn')
    return model
#Third model: ALL-CNN-C
def nin_cnn(model_input):
     #mlpconv block 1
    x = Conv2D(32, (5, 5), activation='relu',padding='valid')(model_input)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    #mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    #mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)

    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='nin_cnn')
    return model



#ensemble model
conv_pool_cnn_model = conv_pool_cnn(model_input)
all_cnn_model = all_cnn(model_input)
nin_cnn_model = nin_cnn(model_input)

conv_pool_cnn_model.load_weights("weights/conv_pool_cnn.05-0.75.hdf5")
all_cnn_model.load_weights("weights/all_cnn.05-0.05.hdf5")
nin_cnn_model.load_weights('weights/nin_cnn.05-0.12.hdf5')

models=[all_cnn_model,all_cnn_model]

def ensemble(models, model_input):

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input , y , name="ensemble")
    return model

ensemble_model = ensemble(models, model_input)
_ = compile_and_train(ensemble_model,num_epochs=5)
evaluate_error(ensemble_model)