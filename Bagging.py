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
