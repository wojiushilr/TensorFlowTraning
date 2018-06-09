import os
import cv2

#定义目录存储数组
namelist=[]
for filename in os.listdir(r"C:\Users\Rivaille\Desktop\ROkinect\dataset3\li\li_test\front"):              #listdir的参数是文件夹的路径
    #print (type(filename))
    namelist.append(filename)
print(len(namelist))
#循环镜像
for i in range(len(namelist)):

  img = cv2.imread('C:\\Users\\Rivaille\\Desktop\\ROkinect\\dataset3\\li\\li_test\\front\\'+namelist[i]+'',cv2.IMREAD_COLOR)
  #print('C:\\Users\\Rivaille\\Desktop\\ROkinect\\data\\feng\\feng_learn\\temp\\'+namelist[i]+'')
  if img is None:
      continue
  xAxis = cv2.flip(img, 0)
  yAxis = cv2.flip(img, 1)
  xyAxis = cv2.flip(img, -1)

  cv2.imwrite('C:\\Users\\Rivaille\\Desktop\\ROkinect\\dataset3\\li\\li_test\\temp2\\'+namelist[i]+'', yAxis)
  #print('C:\\Users\\Rivaille\\Desktop\\ROkinect\\data\\feng\\feng_learn\\temp2\\'+namelist[i]+'')