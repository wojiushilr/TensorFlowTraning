# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from  PIL import Image
import matplotlib.pyplot as plt 
import numpy as np

cwd='/Users/rivaille/PycharmProjects/TensorFlowTraning/NN/temp/'

filename_queue = tf.train.string_input_producer(["train.tfrecords"]) #读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                   })  #取出包含image和label的feature对象
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [128, 64, 3])
label = tf.cast(features['label'], tf.int32)
print(image)


with tf.Session() as sess: #开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    #print(sess.run(image))
    for i in range(3000):
        example, l = sess.run([image,label])#在会话中取出image和label
        #print("example",example)
        print("l", l)
    coord.request_stop()
    coord.join(threads)


'''
with tf.Session() as sess: #开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    print(sess.run(image))
    for i in range(3000):
        example, l = sess.run([image,label])#在会话中取出image和label
        img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
        print(example, l)
    coord.request_stop()
    coord.join(threads)'''

