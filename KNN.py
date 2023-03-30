from math import sqrt
l1 = np.array(Y_test)
def euclidean(r1,r2):
  d1 = 0.0
  for i in range(len(r1)-1):
    d1 += (r1[i] - r2[i])**2
  return sqrt(d1)

def distance_neighbours(X_train,X_test,k):
  dist = list()
  i = 0
  for tr_r in X_train:
    i+=1
    d = euclidean(X_test,tr_r)
    dist.append((i,d))
  dist.sort(key = lambda tup: tup[1])
  neighbours = list()
  for j in range(k):
    neighbours.append(dist[j][0])
  return neighbours

def prediction(X_train,X_test,k):
  n = list()
  for i in X_test:
    neighbours = distance_neighbours(X_test,i,k)
    p = [l1[z-1] for z in neighbours]
    predict = max(set(p),key = p.count)
    n.append(predict)
  return n

t=np.array(X_train)
t1=np.array(X_test)
predictions=prediction(t,t1,4)
n=len(predictions)
count=0
for i in range(n):
  if predictions[i]==l1[i]:
    count+=1
print("accuracy: ", (count/n)*100)
