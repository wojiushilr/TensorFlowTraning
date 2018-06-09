'''
DNN
'''

from __future__ import print_function
import tensorflow as tf
#import tensorlayer as tl
#import numpy as np
from skimage import data, io, filters


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })


    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 64, 3])
    label = tf.cast(features['label'], tf.int32)
    return img, label


x_train, y_train = read_and_decode("C:\\Users\\Rivaille\\Desktop\\TensorFlow\\train.tfrecords")

batch_train_x, batch_train_y = tf.train.shuffle_batch([x_train, y_train],
                                                    batch_size=50, capacity=2000,
                                                    min_after_dequeue=1000,num_threads=32)

init = tf.global_variables_initializer()#用这个没有warning

with tf.Session() as sess:#用此语句session自动关闭
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(3):
        val, l= sess.run([batch_train_x, batch_train_y])
        #我们也可以根据需要对val， l进行处理
        #l = to_categorical(l, 12)
        #读取图片显示。。。
        print(val.shape, l)
        io.imshow(val[0,:,:,:])
        io.show()
