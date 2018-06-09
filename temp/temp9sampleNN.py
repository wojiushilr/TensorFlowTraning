'''
DNN
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import  matplotlib.pyplot as plt
import os
import os.path
import time
from datetime import datetime

import numpy as np

import tensorflow as tf

LEARNING_RATE=0.01
MAX_STEP=2000

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
    img = tf.reshape(img, [64,128,3])
    img = tf.cast(img, tf.float32) * (1. / 255)- 0.5
    label =features['label']
    #label = tf.reshape(label, [3])
    return img, label

def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
         Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('biases'):
         biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
        with tf.name_scope('Wx_plus_b'):
         Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs







#correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_pred,1))  #在测试阶段，测试准确度计算
#此处label和mnist的例子不同是个shape为【64】的而不是【64，3】，所以此处不需要tf.argmax(label,1)

# 汇总操作
summary_op = tf.summary.merge_all()
init = tf.global_variables_initializer()#用这个没有warning
with tf.Session() as sess:
    ###tensorboard --logdir=C:\Users\Rivaille\Desktop\TensorFlow\TrainNN
    train_dir = 'C:\\Users\\Rivaille\\Desktop\\TensorFlow\\TrainNN'
    # tf.train.SummaryWriter soon be deprecated, use following
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
      writer = tf.train.SummaryWriter(train_dir, sess.graph)
    else: # tensorflow version >= 0.12
      writer = tf.summary.FileWriter(train_dir, sess.graph)
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    trainTimes=[]
    for epoch in range(MAX_STEP):
        with tf.name_scope('inputs'):
         train_img, train_label = read_and_decode("C:\\Users\\Rivaille\\Desktop\\TensorFlow\\train.tfrecords")
         test_img, test_label = read_and_decode("C:\\Users\\Rivaille\\Desktop\\TensorFlow\\test.tfrecords")
        # groups examples into batches randomly
        images_batch, labels_batch = tf.train.shuffle_batch(
            [train_img, train_label], batch_size=64,
            capacity=20000,
            min_after_dequeue=1000)
        #labels_batch = tf.one_hot(labels_batch,3,1,0)注意此处数据不shuffle打乱
        images_batch_test, labels_batch_test = tf.train.batch(
            [test_img, test_label], batch_size=64, capacity=20000, num_threads=32)
        #labels_batch_test = tf.one_hot(labels_batch_test,3,1,0)
        print(train_img,train_label)
        #prediction = sess.run(accuracy)
        #true_count = 0
        #true_count += np.sum(prediction)
        #precision = true_count / 10000
        #Sample model setting
        reshape = tf.reshape(images_batch, [64,-1])
        dim = reshape.get_shape()[1].value
        print(dim)
        l1= add_layer(reshape,dim,156,activation_function=tf.nn.relu)
        #l2= add_layer(l1,1024,256,activation_function=tf.nn.relu)
        y_pred= add_layer(l1,156,3,activation_function=None)
        with tf.variable_scope('loss') as scope:
          loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,labels=labels_batch),name = 'loss')
          tf.summary.scalar(scope.name + '/x_entropy', loss)
        correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int64), train_label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


        #print(tf.trainable_variables())

        train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
        sess.run(train_op)

        if epoch % 50==0:
             # pass it in through the feed_dict
             loss_val = sess.run(loss)
             #print ("EPOCH:",epoch,"LOSS:",loss_val)
             #print ("precision:",epoch,sess.run(correct_sum)/64)
             print ("accuracy:",epoch,sess.run(accuracy))
             trainTimes.append(epoch)
             summary_str = sess.run(summary_op)
             writer.add_summary(summary_str, epoch)
             # 保存当前的模型和权重到 train_dir，global_step 为当前的迭代次数
             checkpoint_path = os.path.join(train_dir, 'model.ckpt')

    '''
    #pic shows
    plt.figure(figsize=(8,4))
    plt.plot(trainTimes,l,label="error",color="red",linewidth=1)
    plt.xlabel("trainTimes")
    plt.ylabel("loss")
    plt.legend()
    plt.show()'''


