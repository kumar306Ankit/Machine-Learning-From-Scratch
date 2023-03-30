class pca:
  def __init__(self,n=None):                         #Constructor for initializng variables for input. 
    self.n = n                                       # n = No. of Components

  def covmat(self,X):  
    X_centered = X - np.mean(X, axis=0)  # Compute the covariance matrix using matrix multiplication
    self.C = X_centered.T.dot(X_centered) / (X.shape[0] - 1)
    self.X = X_centered
    return self.C
  def fit(self):
    covariance_matrix = self.C
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
  def transform(self,X):
      mat = self.covmat(X)
      principal_comp = self.fit()
      return np.array(principal_comp)
