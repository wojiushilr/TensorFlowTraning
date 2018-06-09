from __future__ import print_function
#1：矩阵赋值，读取特定矩阵位置
a = []
for i in range(0, 3):
    tmp = []
    for j in range(0, 3):
        tmp.append(0)
    a.append(tmp)
print (a)

a[0][1]=1
print(a)
print(a[0][:2])#表示a这个矩阵的第1行，从第1个到第3个元素的输出，不包括第3个

'''
#错误的赋初始值写法
tmp = []
for j in range(0, 3):
    tmp.append(0)
a = []
for i in range(0, 3):
    a.append(tmp)
print a
'''
#2：矩阵加维
import numpy as np
x = np.linspace(-1,1,3) #一维
x_data = np.linspace(-1,1,3)[:,np.newaxis] #二维，其中维度不同中括号个数不同
print(x,'\n',x_data)

#矩阵相乘
myList = [([1] ) for i in range(3)]
temp=myList*x_data
print('myList',myList)
print(temp)
#tf的矩阵变量

import tensorflow as tf
xs=tf.placeholder(tf.float32,[None,1])
Weights = tf.Variable(tf.random_normal([1,1]))#查查shape怎么回事
W = tf.matmul(xs,Weights)
init = tf.initialize_all_variables()
print('x_data',x_data)
init = tf.initialize_all_variables()
with tf.Session() as sess:
   sess.run(init)

   print(sess.run(W,feed_dict={xs: x_data}))