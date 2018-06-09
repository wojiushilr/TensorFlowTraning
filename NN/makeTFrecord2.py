import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd='/Users/rivaille/PycharmProjects/TensorFlowTraning/NN/temp/'
classes={'back','front', 'side'} #
writer= tf.python_io.TFRecordWriter("train.tfrecords") #要生成的文件

for index,name in enumerate(classes):
    class_path=cwd+name+'/'
    for img_name in os.listdir(class_path):
        img_path=class_path+img_name #每一个图片的地址

        img=Image.open(img_path)
        img= img.resize((64,128))

        img_raw=img.all_vectores[index]#将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw':  tf.train.Feature(
                int64_list=tf.train.Int64List(value=[img].astype("int64")))
        })) #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串

writer.close()