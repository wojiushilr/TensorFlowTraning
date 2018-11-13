import cv2
import numpy as np
from keras.models import load_model

#保存したモデルを読み込み
model = load_model('mnist_demo.h5')

#画像を読み込み、また前処理を行う；28＊28にリサイズ、Gray-scaleを行う
crop_size = (28, 28)
image = cv2.imread('test_pic.jpg')
img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
img2 = cv2.resize(img, crop_size, interpolation= cv2.INTER_CUBIC)
img2 = (img2.reshape(1, 28, 28, 1)).astype('float32')/255

#認識開始
proba = model.predict_proba(img2, verbose = 0)
predict = model.predict_classes(img2, verbose = 0)

#認識結果表示
for i in range(10):
    print ("Probability of {}:".format(i),round(proba[0][i],3))

print ('Recognition result：',predict[0])

#テストデータ表示
cv2.namedWindow("window")
cv2.imshow("window",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


