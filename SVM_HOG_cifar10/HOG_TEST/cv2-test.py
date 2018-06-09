


import cv2
# windows to display image
cv2.namedWindow("Image")
# read image
image = cv2.imread('./data/li_45_20170421_194702.780.png')
# show image
cv2.imshow("Image", image)
# exit at closing of window
cv2.waitKey(0)
cv2.destroyAllWindows()