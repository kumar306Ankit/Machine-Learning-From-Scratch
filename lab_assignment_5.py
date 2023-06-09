# -*- coding: utf-8 -*-
"""Lab_Assignment_5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1blXlkGqOtG13hsku_Swfo6qg7Kp5Wodr

# Question 1: [Bagging]

## 1.1
Create a dataset with 1000 samples, using the ‘make_moon’s function of sklearn (choose random_state=42, noise=0.3). Perform appropriate preprocessing, train and test split of the dataset. Plot the generated dataset
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

c1,c2 = make_moons(n_samples=1000,noise=0.3, random_state=42)    
dataset,dataset['Output'] = pd.DataFrame(c1),c2                
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

X1

y1

sns.scatterplot(data=dataset, x=0, y=1, hue='Output')

"""## 1.2
Train a simple decision tree classifier from sklearn and plot the decision boundary for the same.
Perform hyperparameter tuning for finding the best value of max_depth of the decision tree. [5
marks]
"""

def MSE(X_train, y_train, X_valid, y_valid, maxdepth):
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import mean_squared_error
  regressor = DecisionTreeClassifier(max_depth=maxdepth)
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_valid)
  np.set_printoptions(precision=2)
  return mean_squared_error(y_valid, y_pred)

import matplotlib.pyplot as plt
score = []
x = np.arange(1,11,1)
for i in x:
  score.append(MSE(X_train, y_train, X_test, y_test ,i))
plt.plot(x,score)
plt.xlabel('Max_depth')
plt.ylabel('MSE')
plt.show()
print("For max_depth {}, error is minimum, that is {}".format(x[6], min(score)))

def Boundary(clf,X,y):
    X2, y2 = np.meshgrid(np.linspace(np.min(X[0]), np.max(X[0]), 60), np.linspace(np.min(X[1]), np.max(X[1]), 60))
    clf.fit(X[[0,1]],y)
    Z = clf.predict(pd.DataFrame(np.c_[X2.ravel(), y2.ravel()],columns = [0,1]))
    Z = Z.reshape(X2.shape)
    plt.contourf(X2, y2, Z)
    sns.scatterplot(data=X[[0,1]],x = 0,y = 1,hue = y)
    return plt.show()

# max_depth 7
from sklearn.tree import DecisionTreeClassifier
Decission_tree_classeifier = DecisionTreeClassifier(max_depth=7)
Decission_tree_classeifier.fit(X_train, y_train)
y_pred_Decission_tree_classeifier = Decission_tree_classeifier.predict(X_test)
print(accuracy_score(y_test,y_pred_Decission_tree_classeifier))
Boundary(Decission_tree_classeifier,X_train,y_train)

"""## 1.3
Train a BaggingClassifier from sklearn, on the same dataset, and plot the decision boundary
obtained.
"""

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
Bg = BaggingClassifier()
Bg = Bg.fit(X_train,y_train) 
y_pred = Bg.predict(X_test)

print(accuracy_score(y_test,y_pred))

Boundary(Bg,X_train,y_train)

"""## 1.4
Train a RandomForest classifier from sklearn and plot its decision boundary. Compare the
models (all 3), their decision boundaries, and their accuracy metrics
"""

from sklearn.ensemble import RandomForestClassifier
Ran = RandomForestClassifier()
Ran = Ran.fit(X_train,y_train)  
y_pred = Ran.predict(X_test)

print(accuracy_score(y_test,y_pred))

Boundary(Ran,X_train,y_train)

"""## 1.5
Vary the number of estimators for the BaggingClassifier and RandomForestClassifier, and
comment on the obtained decision boundaries and their accuracies.
"""

def Bgc(Xtrain,ytrain,Xtest,ytest,i):
  from sklearn.ensemble import BaggingClassifier
  from sklearn.metrics import accuracy_score
  Bg = BaggingClassifier(n_estimators = i)
  Bg = Bg.fit(Xtrain,ytrain) 
  ypred = Bg.predict(Xtest)
  Boundary(Bg,Xtrain,ytrain)
  return accuracy_score(ytest,ypred)

Accuracy = []
arr = np.arange(50,150,10)
for i in range(50,150,10):
  a = Bgc(X_train,y_train,X_test,y_test,i)
  Accuracy.append(a)
plt.plot(arr, Accuracy)
plt.show()

"""RandomForest"""

def rgc(Xtrain,ytrain,Xtest,ytest,i):
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score
  Rg = RandomForestClassifier(n_estimators = i)
  Rg = Rg.fit(Xtrain,ytrain) 
  ypred = Rg.predict(Xtest)
  Boundary(Rg,Xtrain,ytrain)
  return accuracy_score(ytest,ypred)

Accuracy = []
arr = np.arange(50,150,10)
for i in range(50,150,10):
  a = rgc(X_train,y_train,X_test,y_test,i)
  Accuracy.append(a)
plt.plot(arr, Accuracy)
plt.show()

"""## 1.6
Implement a Bagging algorithm from scratch. [20 marks]
Note: The code should be well commented and the role of each function should be mentioned
clearly.
Apply the above scratch bagging algorithm with n_estimators = 10, train it on the same dataset
as above. Summarize how each of the separate trees performed (both numerically and visually).
"""

class bag:
  def __init__(self,n_estimators):
      self.n_estimators = n_estimators
    
  def fit(self,X,y):
    from sklearn.tree import DecisionTreeClassifier
    self.Decission_tree_classeifier1 = DecisionTreeClassifier()
    self.X = X #Xtrain
    self.y = y #ytrain
    self.model_list= [] #list models in we store different training models with different trees
    X = np.array(self.X) 
    y = np.array(self.y)
    i=0
    while(i<self.n_estimators): #apply 10 times a loop and train a model for 10 times on different tress
      indices = np.random.choice(len(self.X),int(1.0*len(self.X)),replace = True)
      X_train1 = X[indices]
      y_train1 = y[indices]
      dct = DecisionTreeClassifier() #importing every time a decision tree classifier to make separate model
      dct.fit(X_train1,y_train1)
      self.model_list.append(dct)
      i=i+1
  def predict(self, X_test):
    self.X_test = np.array(X_test)
    self.array_predictions = []
    i=0
    while(i<len(self.X_test)):
        prediction_list = []
        for j in self.model_list:
            prediction_list.append(j.predict(self.X_test[i].reshape(1, -1))[0]) # prediction every row in every model j
        import statistics
        r_p = statistics.mode(prediction_list) #max(set(prediction_list), key=lambda x: prediction_list.count(x))  # most accurate prediction is max times out which comes in  combination of every tree prediction
        self.array_predictions.append(r_p) # add prediction in prediction array
        s = np.array(self.array_predictions)
        i=i+1
    return s
  def accuracies(self, X_test, y_test):
    self.y_test = np.array(y_test)
    self.X_test = np.array(X_test)
    prediction_accurice_list = []
    for j in self.model_list: 
      prediction_list1 = []
      for i in range(len(self.X_test)): # in j model we predict out come of every column and add in prediction in list
        prediction_list1.append(j.predict(self.X_test[i].reshape(1, -1))[0])
      score = accuracy_score(self.y_test,np.array(prediction_list1)) # checking accuracy of y_test and prediction list and this loop is applied for all models and return prediction list in we accuracy score of every model comes
      prediction_accurice_list.append(score)
    return prediction_accurice_list

import statistics
bagging_= bag(10)     
bagging_.fit(X_train, y_train)
y_bag_pred = bagging_.predict(X_test)
print('Accuracy of BaggingModel from Scratch is->',accuracy_score(y_test,y_bag_pred))
print('Accuracy of each tree is ->',bagging_.accuracies(X_test,y_test))

"""# Question 2: [Boosting]

Note: For installing XgBoost write the following command in one of the colab cell. !pip install xgboost For installing LightGBM write the following !pip install lightgbm Using the same dataset as in question 1,
"""

!pip install xgboost

!pip install lightgbm

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

"""## 2.1
1. Train a AdaBoost Model.
"""

from sklearn.ensemble import AdaBoostClassifier
AdaClf = AdaBoostClassifier()
AdaClf.fit(X_train,y_train)
y_AdaClf_pred = AdaClf.predict(X_test)
y_AdaClf_pred

"""## 2.2
Train a XGBoost Model in which subsample=0.7
"""

import xgboost as xgb
Model_xg = xgb.XGBClassifier(subsample=0.7) 
Model_xg.fit(X_train,y_train)
y_xgb_pred = Model_xg.predict(X_test)
y_xgb_pred

"""## 2.3
Print the accuracy on the training set and test set.

### AdaBoost

Training set
"""

print(AdaClf.score(X_train,y_train))

"""Test set"""

print(AdaClf.score(X_test,y_test))

"""### XGboost

Traing Set
"""

print(Model_xg.score(X_train,y_train))

"""Test Set"""

print(Model_xg.score(X_test,y_test))



"""## 2.4
Train a LightGBM model and choose different values for num_leaves
"""

import lightgbm as lgb
Model_lgb = lgb.LGBMClassifier()
Model_lgb.fit(X_train, y_train)
y_lgb_pred = Model_xg.predict(X_test)
y_lgb_pred

def lg(X_train, y_train, X_valid, y_valid, numleaves):
  import lightgbm as lgb
  lg_ = lgb.LGBMClassifier(num_leaves = numleaves)
  lg_.fit(X_train, y_train)
  s = lg_.score(X_test,y_test)
  return s

Accuracy = []
arr = np.arange(2,30,2)
for i in range(2,30,2):
  s_ = lg(X_train,y_train,X_test,y_test,i)
  Accuracy.append(s_)
plt.plot(arr,Accuracy,label='test_score')
plt.xlabel('num_leaves')
plt.ylabel('Score')
plt.show()

best_num_leaves_value = 2+ 2*(Accuracy.index(max(Accuracy)))
best_num_leaves_value

"""best_num_leaves_value = 6 at which best score comes

## 2.5
Analyze the relation between max_depth and num_leaves, and check for which value the model starts overfitting
"""

def Fun_LGB(X_train, y_train, X_valid, y_valid,maxdepth, numleaves):
  import lightgbm as lgb
  lg_ = lgb.LGBMClassifier(num_leaves = numleaves,max_depth= maxdepth)
  lg_.fit(X_train, y_train)
  s_train = lg_.score(X_train,y_train)
  s_test = lg_.score(X_test,y_test)
  return s_train,s_test

"""Max_Depth"""

train_score = []
test_score = []
arr1 = np.arange(1,20)
for i in range(1,20):
  s_train,s_test = Fun_LGB(X_train, y_train, X_test, y_test,i,None)
  train_score.append(s_train)
  test_score.append(s_test)
plt.plot(arr1,train_score,label='train_score')
plt.plot(arr1,test_score,label='test_score')
plt.xlabel('max_depth')
plt.ylabel('Score')
plt.grid()
plt.legend()
plt.show()

"""num_leaves"""

train_score = []
test_score = []
arr2 = np.arange(2,20)
for i in range(2,20):
  s_train,s_test = Fun_LGB(X_train, y_train, X_test, y_test,None,i)
  train_score.append(s_train)
  test_score.append(s_test)
plt.plot(arr2,train_score,label='train_score')
plt.plot(arr2,test_score,label='test_score')
plt.xlabel('num_leaves')
plt.ylabel('Score')
plt.grid()
plt.legend()
plt.show()

"""The graphs show that at max depth = 3 and num leaves = 5, the model begins to overfit as the differences in accuracies between training and testing become substantial.

## 2.6
"""

from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
parameters = {'max_depth': arr1, 'num_leaves': arr2}
model = LGBMClassifier()
Best_parameter = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
Best_parameter.fit(X_train, y_train)
print("Best Parameters are:", Best_parameter.best_params_)

"""## 2.7
Plot the decision boundaries for all the 3 models and compare their performance

AdaBoost
"""

Boundary(AdaClf,X_train,y_train)

"""XGBoost"""

Boundary(Model_xg,X_train,y_train)

"""LightGB"""

Boundary(Model_lgb,X_train,y_train)

from sklearn.metrics import accuracy_score

print('Accuracy of AdaBoost Model :',accuracy_score(y_test,y_AdaClf_pred))

print('Accuracy of XGBoost Model :',accuracy_score(y_test,y_xgb_pred ))

print('Accuracy of LightGBM Model :',accuracy_score(y_test,y_lgb_pred))

"""# Question 3:

## 3.1
Train a Bayes classification model on the above dataset, (using sklearn)(tune the hyperparameters accordingly)
"""

from sklearn.naive_bayes import GaussianNB
Bayes_Clf = GaussianNB()
Bayes_Clf.fit(X_train,y_train)
y_bc_pred = Bayes_Clf.predict(X_test)
Bayes_Clf.score(X_test,y_test)

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
parameters = {'priors': [None, [0.25, 0.75], [0.5, 0.5], [0.75, 0.25]],'var_smoothing': [1e-1, 1e-4, 1e-7, 1e-10, 1e-13]}
Bayes_Clf = GaussianNB()
Best_parameter = GridSearchCV(estimator=Bayes_Clf, param_grid=parameters, cv=5)
Best_parameter.fit(X_train, y_train)
print("Best Parameters are:", Best_parameter.best_params_)

"""## 3.2
From all the above trained models, choose any 3 models of your choice (which are giving good accuracy). Group them along with the trained Bayes Classification model, in a
VotingClassifer from sklearn. Train the VotingClassfier again. And compare its performance with the models which were individually trained.

The 3 models choosen are :DecisionTreeClassifier for max_depth=7,Adaboost,lightgbm they were choosen according to their Accuracies.
"""

from sklearn.ensemble import VotingClassifier
combined_classifier = VotingClassifier(estimators=[('GaussianNB',Bayes_Clf),('DecisionTreeClassifier',Decission_tree_classeifier), ('AdaBoost',AdaClf), ('lightgbm',Model_lgb)],voting='hard')
combined_classifier.fit(X_train, y_train)
y_com_pred = combined_classifier.predict(X_test)
combined_classifier.fit(X_train, y_train)
y_com_pred = combined_classifier.predict(X_test)

print('Accuracy of all Combined Model is->',accuracy_score(y_test,y_com_pred))

print('Accuracy of DecisionTree Model is->',accuracy_score(y_test,y_pred_Decission_tree_classeifier))

print('Accuracy of AdaBoost Model is->',accuracy_score(y_test,y_AdaClf_pred))

print('Accuracy of lightgbm Model is->',accuracy_score(y_test,y_lgb_pred))

print('Accuracy of Bayes Model is->',accuracy_score(y_test,y_bc_pred))