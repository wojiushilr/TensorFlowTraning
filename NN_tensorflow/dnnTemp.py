'''
DNN
'''

from __future__ import print_function
import tensorflow as tf
#import tensorlayer as tl
import numpy as np


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [128, 64, 3])#3指的是3通道

    label = tf.cast(features['label'], tf.int32)

    return img, label

train_img, train_label = read_and_decode("C:\\Users\\Rivaille\\Desktop\\TensorFlow\\train.tfrecords")
test_img, test_label = read_and_decode("C:\\Users\\Rivaille\\Desktop\\TensorFlow\\test.tfrecords")



batch_train_img, batch_train_label = tf.train.shuffle_batch([train_img, train_label],
                                                    batch_size=50, capacity=2000,
                                                    min_after_dequeue=1000,num_threads=32)

batch_test_img, batch_test_label = tf.train.shuffle_batch([test_img, test_label],
        batch_size=50, capacity=2000, min_after_dequeue=1000,num_threads=32)

print(type(batch_test_label))
#####################################
train_labels = tf.one_hot(train_label,3,1,0)
test_labels = tf.one_hot(test_label,3,1,0)
#####################################

#batch_train_img = np.array(batch_train_img)
#batch_train_label = np.array(batch_train_label)
#batch_test_img =np.array(batch_test_img)
#batch_test_label =np.array(batch_test_label)


# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 8192 # data input (img shape: 64*128)
n_output = 3 # total classes (0-9 digits)


# define placeholder for inputs to network
xs = batch_train_img#输入值
ys = batch_train_label#实际值


#model create
def multilayer_perceptron(inputs, in_size, out_size,activation_function=None):
    # Hidden layer with RELU activation
    weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([50,out_size])+0.1)
    Wx_plus_b = tf.matmul(tf.expand_dims(inputs,1),tf.expand_dims(weights))+ biases
    if activation_function is None:
        out_layer= Wx_plus_b
    else:
        out_layer = activation_function(Wx_plus_b)
    return out_layer

l1 = multilayer_perceptron(xs,8192,256,activation_function=tf.nn.relu)
prediction = multilayer_perceptron(l1,256,3,activation_function=None)
#define loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=1))
#train
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(ys,1), tf.argmax(prediction,1))   #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                #多个批次的准确度均值



init = tf.global_variables_initializer()#用这个没有warning

with tf.Session() as sess:#用此语句session自动关闭
     sess.run(init)
     threads = tf.train.start_queue_runners(sess=sess)
     for i in range(3):               #训练阶段，迭代1000次
        #xs,ys=sess.run([batch_train_img,batch_train_label])
        sess.run(train_step)   #执行训练
        if(i%100==0):
            #每训练100次，测试一次
            #xs,ys=sess.run([batch_train_img,batch_train_label])
            print ("accuracy:",sess.run(accuracy))






