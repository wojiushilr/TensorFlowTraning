import cv2
import numpy as np
from keras.models import load_model
#
model = load_model('mnist_demo.h5')
#

crop_size = (28, 28)
image = cv2.imread('test_pic.jpg')
img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)


#
img2 = cv2.resize(img, crop_size, interpolation= cv2.INTER_CUBIC)





#
img2 = (img2.reshape(1, 28, 28, 1)).astype('float32')/255

proba = model.predict_proba(img2, verbose = 0)
predict = model.predict_classes(img2, verbose = 0)

print ('proba：')
print (proba[0])

print ('result：')
print (predict[0])

#
cv2.namedWindow("window")
cv2.imshow("window",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


