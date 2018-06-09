# -*- coding: utf-8 -*-

'''
DNN
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import time
from datetime import datetime

import numpy as np

import tensorflow as tf



def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename],num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128,64,3])#64.64.128.3
    img = tf.cast(img, tf.float32) * (1. / 255)-0.5

    label =features['label']#64
    #label = tf.reshape(label, [3])
    return img, label




#数据解码并读取
train_img, train_label = read_and_decode("train.tfrecords")
test_img, test_label = read_and_decode("test.tfrecords")

print ("train_label",train_label)
print ("train_img",train_img)

#try to read a tensor
'''
with tf.Session() as sess:
    #开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    #print(sess.run(train_label))
    for i in range(3):
        example, l = sess.run([train_img,train_label])#在会话中取出image和label
        print("example",example)
        print("l", l)
    coord.request_stop()
    coord.join(threads)'''


for i in range(2):

    ##############################################################################3
    # groups examples into batches randomly
    images_batch, labels_batch = tf.train.shuffle_batch(
        [train_img, train_label], batch_size=64,
        capacity=20000,
        min_after_dequeue=1000)
    #labels_batch = tf.one_hot(labels_batch,3,1,0)
    #print (labels_batch)

    images_batch_test, labels_batch_test = tf.train.batch(
        [test_img, test_label], batch_size=64, capacity=20000, num_threads=32)
    #labels_batch_test = tf.one_hot(labels_batch_test,3,1,0)

    labels_batch = tf.reshape(labels_batch,[64])

    print ("images_batch:",images_batch)
    print ("labels_batch:",labels_batch)


    with tf.Session() as sess: #开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        example,l = sess.run([images_batch,labels_batch])
        #print ("1,",example)
        print ("epoch{num} label length{ll}:".format(num=i,ll=len(l)))
        print ("epoch{num} label{l}".format(num=i,l=l) )
        coord.request_stop()
        coord.join(threads)
