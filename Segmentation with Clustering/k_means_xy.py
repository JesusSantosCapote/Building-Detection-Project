from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os


actual_path = os.getcwd()
data_path = os.path.join(actual_path, 'data', 'estadio_guillermon_moncada.jpg')

image = Image.open(data_path)

# Convert the image to a matrix of RGB values
image = np.array(image.convert('RGB'))

height, width, _ = image.shape

features = np.zeros((height * width, 5))

# Fill the features with the RGB values and XY coordinates of the pixels of the image 
for i in range(height):
    for j in range(width):
        index = i * width + j
        features[index, 0] = image[i, j, 0] 
        features[index, 1] = image[i, j, 1]
        features[index, 2] = image[i, j, 2]
        features[index, 3] = i
        features[index, 4] = j

n_clusters = 10

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(features)

# Get the cluster labels assigned to each pixel
labels = kmeans.labels_

# Create a new image with the colors of the centroids of each cluster
new_image = np.zeros_like(image)
for i in range(height):
    for j in range(width):
        index = i * width + j
        label = labels[index]
        new_image[i, j, 0] = kmeans.cluster_centers_[label, 0]
        new_image[i, j, 1] = kmeans.cluster_centers_[label, 1]
        new_image[i, j, 2] = kmeans.cluster_centers_[label, 2]

plt.imshow(new_image)
plt.show()