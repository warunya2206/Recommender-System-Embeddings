#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances

closest, min_dist = pairwise_distances_argmin_min(X, clusterer.cluster_centers_)

# Distance of each point to the 9 clusters
X_np = np.array(X)
centroids = clusterer.cluster_centers_
distance = []
for i in X_np:
    data_point = np.array(i).reshape(1,-1)
    distance_to_point = pairwise_distances(data_point, centroids)
    distance.extend(distance_to_point)

distance_all_centroids_df = pd.DataFrame(distance)

