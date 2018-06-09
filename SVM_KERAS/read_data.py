import os
from PIL import Image
import numpy as np
def load_train_data():
    data0 = np.empty((778,24576),dtype="float32")
    label0 = np.empty((778,),dtype="int")
    imgs0 = os.listdir("C:\\Users\\Rivaille\\Desktop\\dataset3\\feng_1\\feng_learn\\0")

    data1 = np.empty((480,24576),dtype="float32")
    label1 = np.empty((480,),dtype="int")
    imgs1 = os.listdir("C:\\Users\\Rivaille\\Desktop\\dataset3\\feng_1\\feng_learn\\1")

    num0 = len(imgs0)
    num1 = len(imgs1)

    for i in range(num0):
        img0 = Image.open("C:\\Users\\Rivaille\\Desktop\\dataset3\\feng_1\\feng_learn\\0\\"+imgs0[i])
        arr = np.asarray(img0,dtype="float32")
        print("arr",arr.shape)
        data0[i] = arr
        label0[i] = int(0)

    data0 /= np.max(data0)
    data0 -= np.mean(data0)

    for i in range(num1):
        img1 = Image.open("C:\\Users\\Rivaille\\Desktop\\dataset3\\feng_1\\feng_learn\\1\\"+imgs1[i])
        arr = np.asarray(img1,dtype="float32")
        data1[i,:,:,:]= arr
        label1[i] = int(1)

    data1 /= np.max(data0)
    data1 -= np.mean(data0)

    return data0,label0,data1,label1

def load_test_data():
    data0 = np.empty((202,24576),dtype="float32")
    label0 = np.empty((202,),dtype="int")
    imgs0 = os.listdir("C:\\Users\\Rivaille\\Desktop\\dataset3\\feng_1\\feng_test\\0")

    data1 = np.empty((118,24576),dtype="float32")
    label1 = np.empty((118,),dtype="int")
    imgs1 = os.listdir("C:\\Users\\Rivaille\\Desktop\\dataset3\\feng_1\\feng_test\\1")

    num0 = len(imgs0)
    num1 = len(imgs1)

    for i in range(num0):
        img0 = Image.open("C:\\Users\\Rivaille\\Desktop\\dataset3\\feng_1\\feng_test\\0\\"+imgs0[i])
        arr = np.asarray(img0,dtype="float32")
        data0[i,:,:,:]= arr
        label0[i] = int(0)

    data0 /= np.max(data0)
    data0 -= np.mean(data0)

    for i in range(num1):
        img1 = Image.open("C:\\Users\\Rivaille\\Desktop\\dataset3\\feng_1\\feng_test\\1\\"+imgs1[i])
        arr = np.asarray(img1,dtype="float32")
        data1[i,:,:,:]= arr
        label1[i] = int(1)

    data1 /= np.max(data0)
    data1 -= np.mean(data0)
    return data0,label0,data1,label1

data0,label0,data1,label1 = load_train_data()
print(len(data0),len(label0),len(data1),len(label1))
print(data0.shape,data1.shape)
print(label0.shape)
#print(label1)
