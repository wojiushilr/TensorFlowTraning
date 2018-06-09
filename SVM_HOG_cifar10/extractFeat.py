
# Import the functions to calculate feature descriptions
from skimage.feature import hog
import numpy as np
from sklearn.externals import joblib
# To read image file and save image feature descriptions
import os
import time
import glob
import pickle
from config import *
import matplotlib.pyplot as plt


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding='latin1')
    fo.close()
    return dict

def getData(filePath):
    TrainData = []
    for childDir in os.listdir(filePath):
        print(childDir)
        if childDir != 'test_batch':
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
            print("data",f)
            train = np.reshape(data["data"], (10000, 3, 32 * 32))
            labels = np.reshape(data['labels'], (10000, 1))
            fileNames = np.reshape(data['filenames'], (10000, 1))
            datalebels = zip(train, labels, fileNames)
            TrainData.extend(datalebels)
        else:
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
            test = np.reshape(data['data'], (10000, 3, 32 * 32))
            labels = np.reshape(data['labels'], (10000, 1))
            fileNames = np.reshape(data['filenames'], (10000, 1))
            TestData = zip(test, labels, fileNames)
    return TrainData, TestData
# hog fearure
def getFeat(TrainData, TestData):
    for data in TestData:
        image = np.reshape(data[0].T, (32, 32, 3))
        gray = rgb2gray(image)/255.0
        #print(gray)
        #ft.hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, transform_sqrt=is_transform, feature_vector=is_feature_vector,
        #cells_per_block=cells_per_block, visualise=is_visualise)
        fd = hog(gray, orientations=9, pixels_per_cell=[5,5], transform_sqrt=True, visualise=visualize)
        #print(fd)
        fd = np.concatenate((fd, data[1]))
        filename = list(data[2])
        fd_name = filename[0].split('.')[0]+'.feat'
        fd_path = os.path.join('./data/features/test/', fd_name)
        joblib.dump(fd, fd_path)
    print ("Test features are extracted and saved.")
    for data in TrainData:
        image = np.reshape(data[0].T, (32, 32, 3))
        gray = rgb2gray(image)/255.0
        #print("done")

        fd = hog(gray, orientations=9, pixels_per_cell=[5,5], transform_sqrt=True, visualise=visualize)
        fd = np.concatenate((fd, data[1]))
        filename = list(data[2])
        fd_name = filename[0].split('.')[0]+'.feat'
        fd_path = os.path.join('./data/features/train/', fd_name)
        joblib.dump(fd, fd_path)
    print ("Train features are extracted and saved.")

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

if __name__ == '__main__':
    print(orientations)
    t0 = time.time()
    filePath = r"D:\\dataset\\cifar-10-batches-py"
    TrainData, TestData = getData(filePath)
    print("ok",np.array(TrainData).shape)
    print(np.array(TrainData))
    getFeat(TrainData, TestData)
    t1 = time.time()
    print ("Features are extracted and saved.")
    print ('The cast of time is:%f'%(t1-t0))




