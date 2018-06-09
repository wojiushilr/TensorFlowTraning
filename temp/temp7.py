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
    img = tf.reshape(img, [128,64,3])
    img = tf.cast(img, tf.float32) * (1. / 255)

    label =features['label']
    #label = tf.reshape(label, [-1])
    return img, label

#数据解码并读取
train_img, train_label = read_and_decode("C:\\Users\\Rivaille\\Desktop\\TensorFlow\\train.tfrecords")
test_img, test_label = read_and_decode("C:\\Users\\Rivaille\\Desktop\\TensorFlow\\test.tfrecords")

####################数据读取#################################3
# groups examples into batches randomly
images_batch, labels_batch = tf.train.shuffle_batch(
    [train_img, train_label], batch_size=64,
    capacity=2000,
    min_after_dequeue=1000)
#labels_batch = tf.one_hot(labels_batch,3,1,0)

images_batch_test, labels_batch_test = tf.train.shuffle_batch(
    [test_img, test_label], batch_size=64,
    capacity=2000,
    min_after_dequeue=1000)
#labels_batch_test = tf.one_hot(labels_batch_test,3,1,0)

labels_batch = tf.reshape(labels_batch,[64])

print(images_batch)
print(labels_batch)


#参数设定
TRAIN = True
LEARNING_RATE=0.1
MAX_STEP=20000
###tensorboard --logdir=C:\Users\Rivaille\Desktop\TensorFlow\train


# 用 get_variable 在 CPU 上定义常量
def variable_on_cpu(name, shape, initializer = tf.constant_initializer(0.1)):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer = initializer,
                              dtype = dtype)
    return var

 # 用 get_variable 在 CPU 上定义变量
def variables(name, shape, stddev):
    dtype = tf.float32
    var = variable_on_cpu(name, shape,
                          tf.truncated_normal_initializer(stddev = stddev,
                                                          dtype = dtype))
    return var

# 定义网络结构
def inference(images):


    with tf.variable_scope('local3') as scope:
        # 第一层全连接
        reshape = tf.reshape(images, [64,-1])
        print(reshape)
        weights = variables('weights', shape=[24576,3], stddev=0.004)
        biases = variable_on_cpu('biases', [3])
        # ReLu 激活函数
        local3 = tf.nn.relu(tf.matmul(reshape, weights)+biases,
                            name = scope.name)
        # 柱状图总结 local3
        tf.summary.histogram(scope.name + '/activations', local3)
        return local3
    '''
    with tf.variable_scope('local4') as scope:
        # 第二层全连接
        weights = variables('weights', shape=[384,192], stddev=0.004)
        biases = variable_on_cpu('biases', [192])
        local4 = tf.nn.relu(tf.matmul(local3, weights)+biases,
                            name = scope.name)
        tf.summary.histogram(scope.name + '/activations', local4)

    with tf.variable_scope('softmax_linear') as scope:
        # softmax 层，实际上不是严格的 softmax ，真正的 softmax 在损失层
        weights = variables('weights', [192, 3], stddev=1/192.0)
        biases = variable_on_cpu('biases', [3])
        softmax_linear = tf.add(tf.matmul(local4, weights), biases,
                                name = scope.name)
'''


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels, tf.int64)
        # 交叉熵损失，至于为什么是这个函数，后面会说明。
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                            (logits=logits, labels=labels, name='cross_entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        tf.summary.scalar(scope.name + '/x_entropy', loss)

    return loss



###########################################################

def train():
    # global_step
    global_step = tf.Variable(0, name = 'global_step', trainable=False)
    # cifar10 数据文件夹
    #data_dir = '/home/your_name/TensorFlow/cifar10/data/cifar-10-batches-bin/'
    # 训练时的日志logs文件，没有这个目录要先建一个
    train_dir = 'C:\\Users\\Rivaille\\Desktop\\TensorFlow\\train'
    # 加载 images，labels
    images=images_batch
    labels=labels_batch

    # 求 loss
    loss = losses(inference(images), labels)
    # 设置优化算法，这里用 SGD 随机梯度下降法，恒定学习率
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    # global_step 用来设置初始化
    train_op = optimizer.minimize(loss, global_step = global_step)
    # 保存操作
    saver = tf.train.Saver(tf.all_variables())
    # 汇总操作
    summary_op = tf.summary.merge_all()
    # 初始化方式是初始化所有变量
    init = tf.initialize_all_variables()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    '''
    config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
                inter_op_parallelism_threads = 1,
                intra_op_parallelism_threads = 1,
                log_device_placement=True)'''
    # 占用 GPU 的 20% 资源
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    # 设置会话模式，用 InteractiveSession 可交互的会话，逼格高
    sess = tf.InteractiveSession(config=config)
    # 运行初始化
    sess.run(init)

    # 设置多线程协调器
    coord = tf.train.Coordinator()
    # 开始 Queue Runners (队列运行器)
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    # 把汇总写进 train_dir，注意此处还没有运行
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    # 开始训练过程
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            start_time = time.time()
            # 在会话中运行 loss
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            # 确认收敛
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 1== 0:
                # 本小节代码设置一些花哨的打印格式，可以不用管
                num_examples_per_step = 64
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

            if step % 10 == 0:
                # 运行汇总操作， 写入汇总
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 100 == 0 or (step + 1) == MAX_STEP:
                # 保存当前的模型和权重到 train_dir，global_step 为当前的迭代次数
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)

    sess.close()


def evaluate():

    #data_dir = '/home/your_name/TensorFlow/cifar10/data/cifar-10-batches-bin/'
    train_dir = 'C:\\Users\\Rivaille\\Desktop\\TensorFlow\\train'
    images= images_batch_test
    labels= labels_batch_test

    logits = inference(images)
    saver = tf.train.Saver(tf.all_variables())

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    # 加载模型参数
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, os.path.join(train_dir, ckpt_name))
        print('Loading success, global_step is %s' % global_step)


    try:
        # 对比分类结果，至于为什么用这个函数，后面详谈
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        true_count = 0
        step = 0
        while step < 157:
            if coord.should_stop():
                break
            predictions = sess.run(top_k_op)
            true_count += np.sum(predictions)
            step += 1

        precision = true_count / 10000
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    except tf.errors.OutOfRangeError:
        coord.request_stop()
    finally:
        coord.request_stop()
        coord.join(threads)

    sess.close()

if __name__ == '__main__':

    if TRAIN:
        train ()
    else:
        evaluate()