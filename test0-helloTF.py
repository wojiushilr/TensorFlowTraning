import tensorflow as tf
hello=tf.constant('Hello TensorFlow!')
sess =tf.Session()
print(sess.run(hello))


a=tf.constant(10)
b=tf.constant(20)
print(sess.run(a+b))

import tensorflow as tf
print(tf.__version__)
print(tf.__path__)