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
