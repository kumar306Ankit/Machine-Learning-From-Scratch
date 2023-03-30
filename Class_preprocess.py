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
