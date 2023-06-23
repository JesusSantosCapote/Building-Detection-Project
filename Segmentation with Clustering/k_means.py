import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

actual_path = os.getcwd()
data_path = os.path.join(actual_path, 'data', 'estadio_guillermon_moncada.jpg')

image_1 = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)

vector_1 = image_1.reshape((-1,3))
kmeans_1 = KMeans(n_clusters=20, random_state = 0, n_init=20).fit(vector_1)
c = np.uint8(kmeans_1.cluster_centers_)
seg_data= c[kmeans_1.labels_.flatten()]
seg_image = seg_data.reshape((image_1.shape))
plt.imshow(seg_image)
plt.show()