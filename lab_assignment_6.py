# -*- coding: utf-8 -*-
"""Lab_Assignment_6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_mYenJQ52wsUjyf4PTGdbNv2jOJs_QsR

# Question 1
K-Means clustering is an unsupervised learning algorithm which groups the unlabeled dataset into
different clusters. [30]

You have been given the Glass Classification Dataset and the details about the dataset is given in the link.
Do the pre-processing of the data before performing the following
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/content/glass.data',header=None)

dataset

"""preprocess"""

class preprocess:
  def __init__(self,data):
    self.data = data
  def nullvalue(self):
    from sklearn.impute import SimpleImputer
    for i in range(len(self.data.columns)): 
        if self.data.iloc[:,i].dtype==object:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            imputer.fit(self.data.iloc[:,i:i+1])
            self.data.iloc[:,i:i+1] = imputer.transform(self.data.iloc[:,i:i+1])
        else:
          imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
          imputer.fit(self.data.iloc[:,i:i+1])
          self.data.iloc[:,i:i+1] = imputer.transform(self.data.iloc[:,i:i+1])
  def encoding(self):
    from sklearn.preprocessing import LabelEncoder
    for i in self.data.columns: 
        if self.data[i].dtype==object:
            le = LabelEncoder()
            self.data[i] = le.fit_transform(self.data[i])
  def split(self,train,test):
    from sklearn.model_selection import train_test_split
    self.train = train
    self.X = self.data.iloc[:,:-1]
    self.y = self.data.iloc[:,-1]
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42, train_size = self.train)
    return self.X,self.y,self.X_train, self.X_test, self.y_train, self.y_test
  def normalization(self,X1,X2):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X1= sc.fit_transform(X1)
    X2 = sc.transform(X2)
    return X1,X2

D = preprocess(dataset)
D.nullvalue()
D.encoding()
X1,y1,X_train, X_test, y_train, y_test = D.split(0.7,0.3)
X_train, X_test = D.normalization(X_train, X_test)

X1

"""## 1.1
Build a k-means clustering algorithm( can use sklearn library) and implement using the value of k which
you find suitable. Visualize this part by showing the clusters along with the centroids.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X1 = lda.fit_transform(X1, y1)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X1)
y_kmeans

from sklearn.metrics import silhouette_score
silhouette_score(X1, y_kmeans)

import seaborn as sns
import matplotlib.pyplot as plt
plt.scatter(X1[y_kmeans == 0, 0], X1[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X1[y_kmeans == 1, 0], X1[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X1[y_kmeans == 2, 0], X1[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.legend()
plt.show()

"""## 1.2
 Use different values of k and find the Silhouette score and then tell which value of k will be optimal and
why?
"""

S_score  = []
for i in range(2,12):
  from sklearn.cluster import KMeans
  from sklearn.metrics import silhouette_score
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
  y_kmeans = kmeans.fit_predict(X1)
  a = silhouette_score(X1, y_kmeans)
  S_score.append(a)
opitimal_k = S_score.index(max(S_score)) + 2

print("Most optimal value of k is :",opitimal_k)

"""## 1.3
There are few methods to find the optimal k value for k-means algorithm like the Elbow Method . Use
the above method to find the optimal value of k.
"""

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

"""most optimal value of k is 3

## 1.4
Apply bagging with the KNN classifier as the base model. Show results with different values of
K(=1,2,3). Comment on whether the accuracy changes or not after bagging with KNN along with the
proper reason in terms of variance and bias
"""

def bagging_knn(k):
  from sklearn.model_selection import cross_val_score
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.ensemble import BaggingClassifier
  knn = KNeighborsClassifier(n_neighbors=k)
  bagging = BaggingClassifier(knn, n_estimators=10, random_state=42)
  scores = cross_val_score(bagging, X1, y1, cv=10)
  return np.mean(scores)
for i in range(1,4):
  print("accuracy for k =",i,"is", bagging_knn(i))

"""# Question 2
We will use the Olivetti dataset for this question (you can download it from any other source
also including libraries). Flatten and preprocess the data (if required) before starting the tasks. It will
become a 4096 dimensional data with 40 classes, more details are available in the link. Inbuilt functions of
sklearn can not be used for this question (except for functions for auxiliary tasks)
"""

from sklearn.datasets import fetch_olivetti_faces
dataset1 = fetch_olivetti_faces()
flat = dataset1.data
img = dataset1.images

"""## 2.1 and 2.2 
Implement a k-means clustering algorithm from scratch.

Make sure that it should:

i) Be a class which will be able to store the cluster centers. 

ii) Take a value of k from users to give k clusters.

iii) Be able to take initial cluster center points from the user as its initialization. 

iv) Stop iterating when it converges (cluster centers are not changing anymore) or, a maximum
iteration (given as max_iter by user) is reached.
"""

class Kmean_Cluster(object):
    def __init__(self,X3,n_initial_centroids,max_ilteration):
      self.X3 = X3
      self.max_ilteration = max_ilteration
      self.n_initial_centroids = n_initial_centroids
    def inintal_cluster_ponts(self):
      indices = np.random.choice(self.X3.shape[0], size=self.n_initial_centroids, replace=False)
      self.random_points = self.X3[indices]
    def clustering(self):
      self.Dic =  {}
      for i in range(0, self.n_initial_centroids):
        key = f'{i}'
        self.Dic[key]  = [self.random_points[i]]
      def distance(Point1,Point2):
        sum = 0
        for i in range(len(Point1)):
          sum += (Point1[i] - Point2[i])**2
        return np.sqrt(sum)
      def centroid(Point1,Point2,key):
        for i in range(len(Point1)):
          Point1[i] = (Point1[i]*len(self.Dic[f'{key}']) + Point2[i])/(len(self.Dic[f'{key}'])+1)
        return Point1
      for i in range(len(self.X3)):
        m = 0
        l = []
        for j in range(self.n_initial_centroids):
          l.append(distance(self.X3[i],self.random_points[j]))
        b = l.index(min(l))
        C1 = self.random_points[b]
        C2 = centroid(self.random_points[b],self.X3[i],b)
        if m!=self.max_ilteration:
          self.random_points[b] = C2
          m += 1
        elif m == self.max_ilteration:
          break
        c = f'{b}'
        self.Dic[c].append(self.X3[i])
      return self.Dic,self.random_points
    def predict(self,X_test):
      def distance(Point1,Point2):
        sum = 0
        for i in range(2):
          sum += (Point1[i] - Point2[i])**2
        return sum

K = Kmean_Cluster(flat,40,20)
K.inintal_cluster_ponts()
l1,l2= K.clustering()

for i in range(40):
  x = len(l1[f'{i}'])
  print('cluster',i,'has',x,'points')

"""## 2.3
Train the k-means model on Olivetti data with k = 40 and 10 random 4096 dimensional points (in input
range) as initializations. Report the number of points in each cluster.

k = 40
"""

K1 = Kmean_Cluster(flat,40,20)
K1.inintal_cluster_ponts()
datasetK1,cluster_pointsK1  = K1.clustering()

for i in range(40):
  x = len(datasetK1[f'{i}'])
  print('cluster',i,'has',x,'points')

"""k = 10"""

K2 = Kmean_Cluster(flat,10,20)
K2.inintal_cluster_ponts()
datasetK2,cluster_pointsK2  = K2.clustering()

for i in range(10):
  x = len(datasetK2[f'{i}'])
  print('cluster',i,'has',x,'points')

"""## 2.4
Visualize the cluster centers of each cluster as 2-d images of all clusters.
"""

import numpy as np
import matplotlib.pyplot as plt
cluster_centers = cluster_pointsK1 
cluster_images = cluster_centers.reshape((-1, 64, 64))
fig = plt.figure(figsize=(20, 20))
for i in range(len(cluster_images)):
    ax = plt.subplot(8, 5, i+1)
    ax.imshow(cluster_images[i], cmap='gray')
    ax.set_title(f'Cluster {i}')
    ax.axis('off')
plt.subplots_adjust(hspace=0.3, wspace=0.1)
plt.show()

"""## 2.5
Visualize 10 images corresponding to each cluster
"""

import numpy as np
import matplotlib.pyplot as plt
for i in range(40):
  print('For cluster',i)
  l = []
  if len(datasetK1[f'{i}']) >= 10:
      for j in range(10):
         l.append(datasetK1[f'{i}'][j])
  else:
      for j in range(len(datasetK1[f'{i}'])):
         l.append(datasetK1[f'{i}'][j])
  import numpy as np
  import matplotlib.pyplot as plt
  cluster_centers = np.array(l)
  cluster_images = cluster_centers.reshape((-1, 64, 64))
  fig = plt.figure(figsize=(20, 20))
  for i in range(len(cluster_images)):
      ax = plt.subplot(8, 5, i+1)
      ax.imshow(cluster_images[i], cmap='gray')
      ax.set_title(f'image {i}')
      ax.axis('off')
  plt.subplots_adjust(hspace=0.3, wspace=0.1)
  plt.show()

"""## 2.6
Train another k-means model with 10 images from each class as initializations , report the number of
points in each cluster and visualize the cluster centers.

k = 10
"""

K2 = Kmean_Cluster(flat,10,20)
K2.inintal_cluster_ponts()
datasetK2,cluster_pointsK2  = K2.clustering()

for i in range(10):
  x = len(datasetK2[f'{i}'])
  print('cluster',i,'has',x,'points')

import numpy as np
import matplotlib.pyplot as plt
clusterCentersK2 = cluster_pointsK2 
clusterimagesK2 = clusterCentersK2.reshape((-1, 64, 64))
fig = plt.figure(figsize=(20, 20))
for i in range(len(clusterimagesK2)):
    ax = plt.subplot(8, 5, i+1)
    ax.imshow(clusterimagesK2[i], cmap='gray')
    ax.set_title(f'Cluster {i}')
    ax.axis('off')
plt.subplots_adjust(hspace=0.3, wspace=0.1)
plt.show()

"""## 2.7
Visualize 10 images corresponding to each cluster
"""

import numpy as np
import matplotlib.pyplot as plt
for i in range(10):
  print('For cluster',i)
  l = []
  if len(datasetK2[f'{i}']) >= 10:
      for j in range(10):
         l.append(datasetK2[f'{i}'][j])
  else:
      for j in range(len(datasetK2[f'{i}'])):
         l.append(datasetK2[f'{i}'][j])
  import numpy as np
  import matplotlib.pyplot as plt
  cluster_centers = np.array(l)
  cluster_images = cluster_centers.reshape((-1, 64, 64))
  fig = plt.figure(figsize=(20, 20))
  for i in range(len(cluster_images)):
      ax = plt.subplot(8, 5, i+1)
      ax.imshow(cluster_images[i], cmap='gray')
      ax.set_title(f'image {i}')
      ax.axis('off')
  plt.subplots_adjust(hspace=0.3, wspace=0.1)
  plt.show()

"""## 2.8
Evaluate Clusters of part c and part f with Sum of Squared Error (SSE) method. Report the scores and
comment on which case is a better clustering.
"""

def distance(Point1,Point2):
    sum = 0
    for i in range(len(Point1)):
      sum += (Point1[i] - Point2[i])**2
    return sum
def SSE(data,Centroid):
  sum = 0
  for i in range(len(data)):
        l = []
        for j in range(len(Centroid)):
          l.append(distance(data[i],Centroid[j]))
        sum += min(l)
  return sum

SSE(flat,cluster_pointsK1)

SSE(flat,cluster_pointsK2)

"""A lower SSE indicates that the data points within each cluster are more tightly packed around their centroid, indicating a better clustering solution.

# Question 3

## 3.1
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class preprocess:
  def __init__(self,data):
    self.data = data
  def nullvalue(self):
    from sklearn.impute import SimpleImputer
    for i in range(len(self.data.columns)): 
        if self.data.iloc[:,i].dtype==object:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            imputer.fit(self.data.iloc[:,i:i+1])
            self.data.iloc[:,i:i+1] = imputer.transform(self.data.iloc[:,i:i+1])
        else:
          imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
          imputer.fit(self.data.iloc[:,i:i+1])
          self.data.iloc[:,i:i+1] = imputer.transform(self.data.iloc[:,i:i+1])
  def encoding(self):
    from sklearn.preprocessing import LabelEncoder
    for i in self.data.columns: 
        if self.data[i].dtype==object:
            le = LabelEncoder()
            self.data[i] = le.fit_transform(self.data[i])
  def split(self,train,test):
    from sklearn.model_selection import train_test_split
    self.train = train
    self.X = self.data.iloc[:,:-1]
    self.y = self.data.iloc[:,-1]
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42, train_size = self.train)
    return self.X,self.y,self.X_train, self.X_test, self.y_train, self.y_test
  def normalization(self,X1,X2):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X1= sc.fit_transform(X1)
    X2 = sc.transform(X2)
    return X1,X2

dataset = pd.read_csv('/content/Wholesale customers data.csv')
dataset

df = pd.DataFrame(dataset)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc1 = MinMaxScaler()
sc2 = StandardScaler()
#dataset = sc1.fit_transform(dataset)
dataset = sc1.fit_transform(dataset)
dataset

"""## 3.2"""

covariance_matrix = np.cov(dataset, rowvar=False)
f1,f2=-1,-1
max=-1
i,j=0,0
for i in range(0,8):
  for j in range(i+1,8):
    print(i,j,np.abs(covariance_matrix[i, j]))
    if(max<np.abs(covariance_matrix[i, j])):
      f1=i
      f2=j
      max=np.abs(covariance_matrix[i, j])
      # print(i,j,max)

print("features indices with max covariance are:",f1,f2)
print("covariance associated with them:",max)

# visualisation of these features :
import matplotlib.pyplot as plt
x = dataset[:, f1]
y = dataset[:, f2]
plt.scatter(x, y)
plt.xlabel('Channel')
plt.ylabel('Detergents_Paper')

plt.show()
sns.boxplot(data=df,x=df['Channel'],y=df['Detergents_Paper'])

"""## 3.3
Apply DBSCAN to cluster the data points and visualize the same.
"""

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
dbscan = DBSCAN()
dbscan.fit(dataset)

num_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))
pca = PCA(n_components = 2)
dataset = pca.fit_transform(dataset)
for cluster_id, color in zip(range(num_clusters), colors):
    mask = dbscan.labels_ == cluster_id
    plt.scatter(dataset[mask, 0], dataset[mask, 1], c=color, label=f'Cluster {cluster_id}')
plt.legend()
plt.show()

"""## 3.4
 Apply Kmeans on the same dataset and compare the visualization with DBSCAN. Comment on what you
observe with reason in the report. [
"""

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(dataset)
pca = PCA(n_components = 2)
dataset34 = pca.fit_transform(dataset)

import seaborn as sns
import matplotlib.pyplot as plt
plt.scatter(dataset34[y_kmeans == 0, 0], dataset34[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(dataset34[y_kmeans == 1, 0], dataset34[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(dataset34[y_kmeans == 2, 0], dataset34[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(dataset34[y_kmeans == 3, 0], dataset34[y_kmeans == 3, 1], s = 100, c = 'black', label = 'Cluster 4')
plt.scatter(dataset34[y_kmeans == 4, 0], dataset34[y_kmeans == 4, 1], s = 100, c = 'olive', label = 'Cluster 5')
plt.scatter(dataset34[y_kmeans == 5, 0], dataset34[y_kmeans == 5, 1], s = 100, c = 'brown', label = 'Cluster 6')
plt.legend()
plt.show()

"""## 3.5
Use the make_moons function of sklearn to create a datasat of 2000 points. Add some noise to the
plot, i.e., randomly add data points to the plot with a 20% probability. Apply DBSCAN and KNN to
cluster them and finally compare the plots and comment on which one is better
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
X_, y_ = make_moons(2000, noise=0.2, random_state=42)
y_

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
dbscan1 = DBSCAN()
dbscan1.fit(X_)

clust = dbscan1.labels_
clust
plt.scatter(X_[:,0], X_[:,1], c=clust)
plt.title('DBSCAN Clustering')
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_)
pca = PCA(n_components = 2)
dataset52 = pca.fit_transform(X_)

num_clusters = len(set(kmeans.labels_)) - (1 if -1 in kmeans.labels_ else 0)
colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))
for cluster_id, color in zip(range(num_clusters), colors):
    mask = kmeans.labels_ == cluster_id
    plt.scatter(dataset52[mask, 0], dataset52[mask, 1], c=color, label=f'Cluster {cluster_id}')
plt.legend()
plt.show()