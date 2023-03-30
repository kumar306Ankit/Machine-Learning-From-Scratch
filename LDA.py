class lda:
  def __init__(self,n=None):                         #Constructor for initializng variables for input. 
    self.n = n                                       # n = No. of Components

  def Scaller_withinclass_between_class(self,X,y):
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    means = np.array([np.mean(X[y == c], axis=0) for c in np.unique(y)])
    mean_data = np.array([np.mean(X, axis=0)])
    self.SW = np.zeros((X.shape[1],X.shape[1]))
    self.SB = np.zeros((X.shape[1],X.shape[1]))
    for i in range(num_classes):
      X_centered = np.array(X[y==unique_classes[i]] - np.mean(np.array(X[y==unique_classes[i]]), axis=0))  # Compute the covariance matrix using matrix multiplication
      # print("lklk",X_centered)
      self.SW = self.SW + X_centered.T.dot(X_centered)
      #print(self.SW.shape,X_centered.T.dot(X_centered).shape)
      # print(X_centered.T.dot(X_centered))
      X_centeredB = means[i] - mean_data[0]
      num = len(X[y==unique_classes[i]])  # Compute the covariance matrix using matrix multiplication
      self.SB += num*(X_centeredB.T.dot(X_centeredB))
    self.X = X
    self.y = y
    # print(self.SW)
    return self.SW,self.SB
  def fit(self,X,y):
    import pandas as pd
    A = (np.linalg.inv(self.SW)).dot(self.SB)
    covariance_matrix = A
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix) # Compute the eigenvalues and eigenvectors using numpy.linalg.eig
    self.eigenvalues_dp = [] 
    self.vector = []
    l = []
    for i in range(len(eigenvectors)): # Compute the eigenvalues using the dot product method
        v = eigenvectors[:, i]
        eigenvalue_dp = np.dot(v.T, np.dot(covariance_matrix, v)) / np.dot(v.T, v)
        self.eigenvalues_dp.append(eigenvalue_dp)
        self.vector.append(np.array(v))
        l.append((eigenvalue_dp,v)) 
    l = sorted(l,reverse = True)
    self.eigenvalues = np.array([i[0] for i in l])
    self.eigvects = np.array([i[1] for i in l]).T  
    P = np.array(self.X)
    if self.n is None:                               # If n = None there will be no change in no. of components.
        self.principal_components = P@(self.eigvects)
    else:
        self.principal_components = P@(self.eigvects[:,:self.n])
    return self.principal_components
