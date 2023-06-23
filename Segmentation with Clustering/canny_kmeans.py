import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

actual_path = os.getcwd()
data_path = os.path.join(actual_path, 'data', 'estadio_guillermon_moncada.jpg')

img = cv2.imread(data_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)

edges = cv2.Canny(blur, 30, 150, apertureSize=3)

Z = img.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 10
ret,label,center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
kmeans_result = res.reshape((img.shape))

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(img)
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 50:
        cv2.drawContours(mask, contours, i, (255, 255, 255), -1)

segmented_img = cv2.bitwise_and(kmeans_result, mask)

cv2.imshow("Segmented Image", segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()