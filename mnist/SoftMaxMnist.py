# -*- coding: utf-8 -*-
#以上表示python2.7版本默认编码设置语句
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
#y_actual = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))        #初始化权值W
b = tf.Variable(tf.zeros([10]))            #初始化偏置项b
y_predict = tf.nn.softmax(tf.matmul(x,W) + b)     #加权变换并进行softmax回归，得到预测概率
y_=tf.placeholder("float",[None,10]) #

print('x',x)#x Tensor("Placeholder:0", shape=(?, 784), dtype=float32)
print(y_)   #Tensor("Placeholder_1:0", shape=(?, 10), dtype=float32)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),reduction_indies=1))   #求交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y_predict))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)   #用梯度下降法使得残差最小
print(y_predict,y_)
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_,1))   #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                #多个批次的准确度均值

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    trainTimes=[]
    acc=[]
    for i in range(1000):               #训练阶段，迭代1000次
        batch_xs, batch_ys = mnist.train.next_batch(100)           #按批次训练，每批100行数据
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})   #执行训练
        trainTimes.append(i)
        acc.append(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        if(i%100==0):                  #每训练100次，测试一次
            print ("accuracy:",i,sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    '''graph of accuracy'''

    plt.figure(figsize=(8,4))
    plt.plot(trainTimes,acc,label="error",color="red",linewidth=1)
    plt.xlabel("trainTimes")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()