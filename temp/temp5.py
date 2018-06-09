'''
DNN
'''

from __future__ import print_function
import tensorflow as tf
import  matplotlib.pyplot as plt
#import tensorlayer as tl
import numpy as np


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
    img = tf.cast(img, tf.float32) * (1. / 255)

    label =features['label']
    return img, label

def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in range(batch_size)], [
          fake_label for _ in range(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples: # epoch中的句子下标是否大于所有语料的个数，如果为True,开始新一轮的遍历
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples) # arange函数用于创建等差数组
      np.random.shuffle(perm)  # 打乱
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def fill_feed_dict(data_set, images_pl, labels_pl):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(64,fake_data=False)

  images_pl = images_feed,
  labels_pl =labels_feed,

  return images_pl,labels_pl


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def train(labels_batch,y_pred):
    #y_pred =tf.cast(y_pred,tf.float32)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_batch, logits=y_pred))
    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels_batch-y_pred), reduction_indices=1))
    #loss = -tf.reduce_sum(labels_batch*tf.log(y_pred))
    # for monitoring

    print(tf.trainable_variables())
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    return train_op


###########################################################################################################

train_img, train_label = read_and_decode("C:\\Users\\Rivaille\\Desktop\\TensorFlow\\train.tfrecords")
test_img, test_label = read_and_decode("C:\\Users\\Rivaille\\Desktop\\TensorFlow\\test.tfrecords")

'''
# groups examples into batches randomly
images_batch, labels_batch = tf.train.shuffle_batch(
    [train_img, train_label], batch_size=64,
    capacity=2000,
    min_after_dequeue=1000)'''
#labels_batch = tf.one_hot(labels_batch,3,1,0)注意此处数据不shuffle打乱
images_batch_test, labels_batch_test = tf.train.batch(
    [test_img, test_label], batch_size=64,
    capacity=2000,
    min_after_dequeue=1000)
#labels_batch_test = tf.one_hot(labels_batch_test,3,1,0)
print(test_img,train_label)



'''
# simple model
w = tf.Variable(tf.zeros([24576,3]))
b = tf.Variable(tf.zeros([3]))
print(w)
y_pred =  tf.nn.softmax(tf.matmul(reshape, w)+b)
#labels_batch = tf.cast(labels_batch,tf.float32)
'''


################################
with tf.Session() as sess:
        init = tf.global_variables_initializer()#用这个没有warning
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        trainTimes=[]
        l=[]
        reshape = tf.reshape(train_img, [64,-1])
        dim = reshape.get_shape()[1].value
        l1= add_layer(reshape,dim,256,activation_function=tf.nn.relu)
        l2= add_layer(l1,256,256,activation_function=tf.nn.relu)
        y_pred= add_layer(l1,256,3,activation_function=None)

        for epoch in range(100):
            images_batch,labels_batch=fill_feed_dict(train_img)
            sess.run(train())
            if epoch % 1==0:
             # pass it in through the feed_dict
             loss_val = sess.run(loss)
             print (epoch,loss_val)
             trainTimes.append(epoch)
             l.append(loss_val)


        #pic shows
        plt.figure(figsize=(8,4))
        plt.plot(trainTimes,l,label="error",color="red",linewidth=1)
        plt.xlabel("trainTimes")
        plt.ylabel("loss")
        plt.legend()
        plt.show()


