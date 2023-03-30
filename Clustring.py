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
