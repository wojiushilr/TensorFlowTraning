'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''

from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

max_words = 1000  # vocab大小
batch_size = 32  # mini_batch_size
nb_epoch = 5  # 大循环次数

print('Loading data...')
(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)  # 载入路透社语料<br>#打印
print('train sequences',X_train.shape)
print(len(X_test), 'test sequences')
# 分类数目--原版路透社我记着是10来着，应该是语料用的是大的那个
nb_classes = np.max(y_train) + 1
print(nb_classes, 'classes')

print('Vectorizing sequence data...') # tokenize
tokenizer = Tokenizer(
    nb_words=max_words) # 序列化，取df前1000大<br>#这里有个非常好玩的事， X_train 里面初始存的是wordindex，wordindex是按照词大小来的（应该是，因为直接就给撇了）<br>#所以这个效率上还是很高的<br>#转化的还是binary，默认不是用tfidf
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Convert class vector to binary class matrix (for use with categorical_crossentropy)') # 这个就好理解多了， 编码而已
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print('Building model...')
model = Sequential() # 第一层<br>#Dense就是全连接层
model.add(Dense(512, input_shape=(max_words,)))  # 输入维度, 512==输出维度
model.add(Activation('relu'))  # 激活函数
model.add(Dropout(0.5))  # dropout<br><br>#第二层
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# 损失函数设置、优化函数，衡量标准
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 训练，交叉验证
history = model.fit(X_train, Y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, Y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])