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

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 64, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label


x_train, y_train = read_and_decode("C:\\Users\\Rivaille\\Desktop\\TensorFlow\\train.tfrecords")
x_test, y_test = read_and_decode("C:\\Users\\Rivaille\\Desktop\\TensorFlow\\test.tfrecords")

print("train_x", x_train.shape)
print("train_y", y_train.shape)
print("test_x",  x_test.shape)
print("test_y",  y_test.shape)


batch_train_x, batch_train_y = tf.train.shuffle_batch([x_train, y_train],
                                                    batch_size=50, capacity=2000,
                                                    min_after_dequeue=1000,num_threads=32)

batch_test_x, batch_test_y = tf.train.batch([x_test, y_test],
        batch_size=50, capacity=20000, num_threads=32)
#####################################
train_labels = tf.one_hot(batch_train_y,3,1,0)
test_labels = tf.one_hot(batch_test_y,3,1,0)
#####################################

print("batch_train_x", batch_train_x)
print("batch_train_y", batch_train_y)
print("batch_test_x", batch_test_x)
print("batch_test_y", batch_test_y)
print("train_labels", train_labels)
print("test_labels", test_labels)

# Parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 30
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 8192 # data input (img shape: 64*128)
n_classes = 3 # total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int32, [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer,miniminze the loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

#train
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# Launch the graph
# Initializing the variables
init = tf.initialize_all_variables()
with tf.Session() as sess:
   sess.run(init)
   trainTimes=[]
   acc=[]
   for i in range(50):
       batch_xs, batch_ys = sess.run([batch_train_x, train_labels])
       sess.run(train_step,feed_dict={x: batch_xs,y: batch_ys})
       trainTimes.append(i)
       acc.append(sess.run(accuracy,feed_dict={x: batch_test_x, y: test_labels}))
       if True:
           print("accuracy:",i,sess.run(accuracy,feed_dict={x: batch_test_x,y: test_labels}))
   sess.close()

