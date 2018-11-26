import numpy as np
from PIL import  Image
import os
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate


def read_image(imagefile, dtype=np.float32):
    image = np.array(Image.open(imagefile), dtype=dtype)
    return image


def save_image(image, imagefile, data_format='channel_last'):
    image = np.asarray(image, dtype=np.uint8)
    image = Image.fromarray(image)
    image.save(imagefile)


def cutout(image_origin, mask_size, mask_value='mean'):
    image = np.copy(image_origin)
    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size
    if top < 0:
        top = 0
    if left < 0:
        left = 0
    image[top:bottom, left:right, :].fill(mask_value)
    return image


'''
img = read_image('/Users/rivaille/Desktop/experiment_data/model1/train/0/li_0_20170521_190911.131.png')  # this is a PIL image
print(img.shape)
for i in range(9):
 save_image(cutout(img, img.shape[0] // 3), "cutout{}.png".format(i))
'''


namelist = []
for filename in os.listdir(r"C:\\Users\\USER\Desktop\\data_2\\model2\\train\\7"):  # listdir的参数是文件夹的路径
     # print (type(filename))
     namelist.append(filename)
print(namelist[1])


#循环镜像
for i in range(len(namelist)):

  img = read_image('C:\\Users\\USER\Desktop\\data_2\\model2\\train\\7\\{}'.format(namelist[i]))
  #print('C:\\Users\\Rivaille\\Desktop\\ROkinect\\data\\feng\\feng_learn\\temp\\'+namelist[i]+'')
  if img is None:
      continue
  save_image(cutout(img, img.shape[0] // 3), 'C:\\Users\\USER\Desktop\\data_2\\model2_cutout\\train\\7\\cutout_{}'.format(namelist[i]))
  #print('/Users/rivaille/Desktop/experiment_data/model1/train/0/{}'.format(namelist[i]))
