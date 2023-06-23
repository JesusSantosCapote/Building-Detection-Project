from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import tqdm

actual_path = os.getcwd()
data_path = os.path.join(actual_path, 'data', 'estadio_guillermon_moncada.jpg')

image = Image.open(data_path)

image = np.array(image.convert('RGB'))

height, width, _ = image.shape

features = np.zeros((height * width, 5))

for i in range(height):
    for j in range(width):
        index = i * width + j
        features[index, 0] = image[i, j, 0]
        features[index, 1] = image[i, j, 1]
        features[index, 2] = image[i, j, 2]
        features[index, 3] = i
        features[index, 4] = j

# Try 2 to 30 clusters.
n_clusters = list(range(2, 30 + 1, 1))
kmeans = []
inertias = []
for i in tqdm.trange(len(n_clusters)):
    kmeans.append(KMeans(n_clusters = n_clusters[i], 
                         random_state = 42, n_init='auto'))
    kmeans[-1].fit(features)
    inertias.append(kmeans[-1].inertia_)
plt.figure(figsize = [20, 5])
plt.subplot(1, 2, 1)
plt.plot(n_clusters, inertias, "-o")
plt.xlabel("$k$", fontsize = 14)
plt.ylabel("Inertia", fontsize = 14)
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(n_clusters[:-1], np.diff(inertias), "-o")
plt.xlabel("$k$", fontsize = 14)
plt.ylabel("Change in inertia", fontsize = 14)
plt.grid(True)
plt.show()