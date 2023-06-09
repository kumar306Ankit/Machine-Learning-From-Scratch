# -*- coding: utf-8 -*-
"""Lab_Assignment_4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hu6nmPWypKhnnX_Rh3nVmVo4YdY8pNCl

1

Part 1
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']
df = pd.read_csv('/content/iris.data',names = col_names)
df = pd.DataFrame(df)
df

import matplotlib.pyplot as plt
for i in df.columns:
  column = df[i]
  plt.hist(column,edgecolor='black',bins = 20)
  plt.title("Histogram of Column Values")
  plt.xlabel(i)
  plt.ylabel("Frequency")
  plt.show()

df.describe()

df.isnull().sum()

"""1. Implement a Gaussian Bayes Classifier class from scratch.(You are not allowed to use the inbuilt scikit function, you are only allowed to use numpy and pandas). The classifier class must have 3 variants defined using its constructor, for each of the cases given below.

2. The Gaussian Bayes Classifier class should also have the following function:

  a. Train: Takes x,y (training data) as input and trains the model.

  b. Test: Takes testing data, testing labels as input, and outputs the predictions for every instance in the testing data, and also the accuracy.

  c. Predict: Takes a single data point as input, and outputs the predicted class.

  d. Plot decision boundary: Takes input the training data points, and their labels, and plots the decision boundary of the model with the data points superimposed on it. (Consider only two features while plotting the decision boundary)
"""

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

D = preprocess(df)
D.nullvalue()
D.encoding()
X,y,X_train, X_test, y_train, y_test = D.split(0.7,0.3)
X_train,X_test = D.normalization(X_train,X_test)

class Gaussian:
  def __init__(self, variant):
        self.variant = variant
  def priorProbabilities_train(self,X,y):
    self.X  = X
    self.y = y
    self.n_samples, self.n_features = self.X.shape
    self.classes = np.unique(self.y)
    self.n_classes = len(self.classes)
    pre_probab = []
    for i in range(0,self.n_classes):
      a = list(self.y)
      b = int(a.count(self.classes[i]))
      pre_probab.append(b/len(a))
    self.prior = pre_probab
    return self.n_classes,self.prior

  def var(self):
    if self.variant == 1:
      self.variance = []
      I = np.identity(self.n_features)
      F1 = pd.DataFrame(self.X)
      feature_variance = np.var(F1.iloc[:,1])
      v = feature_variance*I
      for i in range(self.n_classes):
        self.variance.append(v)
    elif self.variant == 2:
      self.variance = []
      F2 = pd.DataFrame(self.X.T)
      v = np.cov(F2)
      for i in range(self.n_classes):
        self.variance.append(v)
    elif self.variant == 3:
      self.variance = []
      data = np.ones((len(self.X),len(self.X[0])+1))
      for i in range(len(self.X)):
        for j in range(len(self.X[0])+1):
          if j < len(self.X[0]):
            data[i][j] = self.X[i][j]
          elif j == len(self.X[0]):
            data[i][j] = np.array(self.y)[i]
      self.Data = pd.DataFrame(data)
      d = dict(tuple(self.Data.groupby(self.Data.columns[-1])))
      self.variance = []
      for i in range(self.n_classes):
        self.DF1 = d[np.unique(self.y)[i]].iloc[:,:-1]
        F3 = pd.DataFrame(np.array(self.DF1).T)
        v = np.cov(F3)
        self.variance.append(v)

  def meanlist(self):
    self.mean = []
    data = np.ones((len(self.X),len(self.X[0])+1))
    for i in range(len(self.X)):
      for j in range(len(self.X[0])+1):
        if j < len(self.X[0]):
          data[i][j] = self.X[i][j]
        elif j == len(self.X[0]):
          data[i][j] = np.array(self.y)[i]
    self.Data = pd.DataFrame(data)
    d = dict(tuple(self.Data.groupby(self.Data.columns[-1])))
    for i in range(self.n_classes):
        self.DF2 = d[np.unique(self.y)[i]].iloc[:,:-1]
        F4 = pd.DataFrame(np.array(self.DF2))
        m1 = []
        for j in F4.columns:
            m = np.mean(F4[j])
            m1.append(m)
        self.mean.append(m1)
    return self.mean

  def likelihood(self,value,class_value):
    self.mean=self.meanlist()
    a = value.T - np.array(self.mean[class_value]).T
    b = value - np.array(self.mean[class_value])
    c = np.linalg.inv(self.variance[class_value])
    # print(a.shape,b.shape,c.shape)
    prob = (-0.5)*(a@c@b) - 0.5*np.log(np.linalg.det(self.variance[class_value])) + np.log(self.prior[class_value])
    return prob

  def predict(self,X_test,k):
    pre = np.ones(len(X_test))
    n,priorProbabilities = self.n_classes,self.prior
    self.var()
    for i in range(len(X_test)):
      probility = []
      l = np.array(X_test[i])
      for j in range(n):
        prob = self.likelihood(l,j)
        probility.append(prob)
      max_index = probility.index(max(probility))
      pre[i] = max_index
    return pre

  def decbound(self,X,y,f1,f2):
      print(X.shape)
      X1, y1 = np.meshgrid(np.linspace(np.min(X[:,f1]), np.max(X[:,f1]), 50), np.linspace(np.min(X[:,f2]), np.max(X[:,f2]), 50))
      self.priorProbabilities_train(X[:,[f1,f2]],y)
      self.var()
      self.meanlist()
      Z = self.predict(np.c_[X1.ravel(), y1.ravel()],self.variant)
      Z = Z.reshape(X1.shape)
      plt.contourf(X1, y1, Z, cmap=plt.cm.RdYlBu,alpha = 0.6)
      X3 =pd.DataFrame(X)
      print(y)
      y = pd.Series(y)
      sns.scatterplot(data=X3[[f1,f2]],x = f1,y = f2,hue = y)
      plt.show()

model = Gaussian(1)
model.decbound(X_train,y_train,0,1)

model = Gaussian(2)
model.decbound(X_train,y_train,0,1)

model = Gaussian(3)
model.decbound(X_train,y_train,0,1)

"""1.4 Perform 5 fold cross validation on the training dataset and report the accuracies on each validation set as well as comment on the generalizability of each model."""

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

"""Variant 1"""

from sklearn.model_selection import KFold
n = 5
k=1
MODEL2 = KFold(n_splits=n, shuffle=True)
accuracies = []
y=pd.DataFrame(y)
for train_index, test_index in MODEL2.split(X):
    X_train, X_test = np.array(X.iloc[train_index,:]), np.array(X.iloc[test_index,:])
    y_train, y_test = np.array(y.iloc[train_index,:]), np.array(y.iloc[test_index,:])
    model1 = Gaussian(k)
    n,priorProbabilities = model1.priorProbabilities_train(X_train,y_train)
    model1.var()
    y_pred1=model1.predict(X_test, k)
    y_test=y_test.reshape(1,-1)[0]
    p = np.array(y_test)
    accuracies.append(((y_pred1 == p).sum())/len(y_test))
# Calculate the mean Accuracy score across all iterations
mean_acc = np.mean(accuracies)
print("Mean Accuracy:", mean_acc)

"""Variant 2"""

from sklearn.model_selection import KFold
n = 5
k=2
MODEL2 = KFold(n_splits=n, shuffle=True)
accuracies = []
y=pd.DataFrame(y)
for train_index, test_index in MODEL2.split(X):
    X_train, X_test = np.array(X.iloc[train_index,:]), np.array(X.iloc[test_index,:])
    y_train, y_test = np.array(y.iloc[train_index,:]), np.array(y.iloc[test_index,:])
    model1 = Gaussian(k)
    n,priorProbabilities = model1.priorProbabilities_train(X_train,y_train)
    model1.var()
    y_pred1=model1.predict(X_test, k)
    y_test=y_test.reshape(1,-1)[0]
    p = np.array(y_test)
    accuracies.append(((y_pred1 == p).sum())/len(y_test))
# Calculate the mean Accuracy score across all iterations
mean_acc = np.mean(accuracies)
print("Mean Accuracy:", mean_acc)

"""Variant 3"""

from sklearn.model_selection import KFold
n = 5
k=3
MODEL2 = KFold(n_splits=n, shuffle=True)
accuracies = []
y=pd.DataFrame(y)
for train_index, test_index in MODEL2.split(X):
    X_train, X_test = np.array(X.iloc[train_index,:]), np.array(X.iloc[test_index,:])
    y_train, y_test = np.array(y.iloc[train_index,:]), np.array(y.iloc[test_index,:])
    model1 = Gaussian(k)
    n,priorProbabilities = model1.priorProbabilities_train(X_train,y_train)
    model1.var()
    y_pred1=model1.predict(X_test, k)
    y_test=y_test.reshape(1,-1)[0]
    p = np.array(y_test)
    accuracies.append(((y_pred1 == p).sum())/len(y_test))
# Calculate the mean Accuracy score across all iterations
mean_acc = np.mean(accuracies)
print("Mean Accuracy:", mean_acc)

"""1.5 Create a synthetic dataset which has 2 features, and data is generated from a circular distribution: 𝑥^2 + 𝑦^2 = 25. The data has 2 classes, points which have distance <=3 have class=1 and points having distance>3 and distance<= 5 have class=2. {Note that the classes are thus, not linearly separable) Train the implemented Gaussian Bayes Classifier(case 3) on such a synthetic dataset, and plot the decision boundary for the same"""

radii = np.random.uniform(low=0, high=5, size=(100,))
angle =np.random.uniform(low=0, high=2*np.pi, size=(100,))
xx1 = radii*np.cos(angle)
xx2 = radii*np.sin(angle)
data = np.concatenate([np.array([xx1]).T,np.array([xx2.T]).T],axis=1)
output = np.array([1 if r<=3 else 2 for r in radii])

model = Gaussian(3)
model.decbound(data,output,0,1)

"""# Question 2"""

import numpy as np
import pandas as pd
# mean is [0,0]
U = [0, 0]
# Covariance matrix is [[3/2 1/2],[1/2 3/2]]
co_variance = [[3/2, 1/2], [1/2, 3/2]]

print(U)

print(co_variance)

# Here U argument specifies the mean of the distribution and co_variance argument specifies the covariance matrix. 1500, indicates the number of samples I want to generate
X = np.random.multivariate_normal(U, co_variance, 1500)

print(X)

"""# Question 2 
## Part 1



"""

Z = pd.DataFrame(X)

Mean = []
for i in range(len(pd.DataFrame(X).iloc[1,:])):
  x = 0
  for j in range(len(pd.DataFrame(X).iloc[:,i])):
    x+= pd.DataFrame(X).iloc[j,i]
    #print(X)
  x = x/len(Z)
  Mean.append(x)
Mean

# calculating X_cenentred = X - mean
for i in range(len(pd.DataFrame(X).iloc[1,:])):
  for j in range(len(pd.DataFrame(X).iloc[:,i])):
    Z.iloc[j,i] -= Mean[i]

covariance_matrix = np.dot(Z.T,Z) / (X.shape[0] - 1)
covariance_matrix

# np.cov() expects the variables to be in the columns of the input array, not in the rows. By transposing X, I ensure that the variables are represented in the columns of the input array, which allows the np.cov() to calculate the covariance matrix rightly
A = X.T
co_var = np.cov(A)
print(co_var)

# import inbuilt function eig from linalg and import linlag from numpy it gives us eigen value and vector
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print("Eigenvectors : \n", eigenvectors,"\n")
print("Eigenvalues : ", eigenvalues)

import pandas
X = pandas.DataFrame(X)

import matplotlib.pyplot as plt
l0 = [i for i in X.iloc[:,0].values]
l1 = [i for i in X.iloc[:,1].values]
plt.plot(l0,l1,'.')
plt.axis('equal') # to make two plots on same plot and both have same scales on axis
plt.plot(eigenvectors[:,0],eigenvectors[:,1],'.')

"""# Question 2 
## Part 2



"""

import numpy as np
import scipy.linalg as sciLin
co_variance1 = np.linalg.inv(co_variance)
sqrt_co_variance = sciLin.sqrtm(co_variance1)

C = np.dot(sqrt_co_variance, X.T)

Y = C.T
Y

B = pd.DataFrame(Y)
B

Mean1 = []
for i in range(len(pd.DataFrame(Y).iloc[1,:])):
  y = 0
  for j in range(len(pd.DataFrame(Y).iloc[:,i])):
    y+= pd.DataFrame(Y).iloc[j,i]
  y = y/len(B)
  Mean1.append(y)
Mean1

# calculating Y_cenentred = Y - mean
for i in range(len(pd.DataFrame(Y).iloc[1,:])):
  for j in range(len(pd.DataFrame(Y).iloc[:,i])):
    B.iloc[j,i] -= Mean[i]

covariance_matrix = np.dot(B.T,B) / (Y.shape[0] - 1)
covariance_matrix

"""## Part 3"""

import numpy as np
import pandas as pd
def sample_points(n):
  points = []
  theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
  x = 5 * np.cos(theta)
  y = 5 * np.sin(theta)
  for i in range(n):
    points.append([np.array(x)[i],np.array(y)[i]])
  return points

P = sample_points(10)
point = pd.DataFrame(P)
X_Points = [i for i in point.iloc[:,0]]
Y_Points = [y for y in point.iloc[:,1]]
print(X_Points)
print(Y_Points)

import matplotlib.pyplot as plt
l0 = [i for i in X.iloc[:,0].values]
l1 = [i for i in X.iloc[:,1].values]
plt.plot(l0,l1,'.')
plt.axis('equal')
plt.plot(X_Points,Y_Points,'.',color = 'red')

mean = np.array(np.mean(point)) 
print(mean)

print(U[0])

X_p = [i- U[0] for i in X_Points]
Y_p = [j- U[1] for j in Y_Points]
euclidean_distance = []
print(X_p)
print(Y_p)
for i in range(len(X_p)):
  x = np.sqrt((X_p[i])**2 +(Y_p[i])**2)
  euclidean_distance.append(x)
euclidean_distance

import matplotlib.pyplot as plt
def bar_plot(distances):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(distances)), distances)
    ax.set_xlabel("Point Index")
    ax.set_ylabel("Euclidean Distance")
    plt.show()
bar_plot(euclidean_distance)

"""## Part4"""

D = np.dot(sqrt_co_variance, np.array(P).T)
E = D.T

E

pointE = pd.DataFrame(E)
X_EPoints = [i for i in pointE.iloc[:,0]]
Y_EPoints = [y for y in pointE.iloc[:,1]]

XE_p = [i- U[0] for i in X_EPoints]
YE_p = [j- U[1] for j in Y_EPoints]
euclidean_distance_P = []
print(XE_p)
print(YE_p)
for i in range(len(XE_p)):
  x = np.sqrt((XE_p[i])**2 +(YE_p[i])**2)
  euclidean_distance_P.append(x)
euclidean_distance_P

bar_plot(euclidean_distance_P)