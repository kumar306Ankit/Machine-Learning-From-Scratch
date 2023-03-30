def gini_func(dataset):   #this function will find the gini impurity of that dataset
  unique_classes, class_counts = np.unique(dataset, return_counts=True)     # give array of unique classes and array of counts of each class
  n=len(dataset)
  prob = class_counts/n
  p_s=0
  for i in range(len(unique_classes)):
    p_s+=prob[i]**2
  gini_cost=1-p_s       
  return(gini_cost)
p1=gini_func(y)     #example
print(p1)

def gini(X, y, feature_name, split_value):
    X_split_left_side = X[X[feature_name] <= split_value][feature_name]
    y_split_left_side = y[X[feature_name] <= split_value]
    X_split_right_side = X[X[feature_name] > split_value][feature_name]
    y_split_right_side = y[X[feature_name] > split_value]
    l1 = len(X_split_left_side)
    l2 = len(X_split_right_side)
    giniProbability = (l1/len(X))*gini_func(y_split_left_side) + (l2/len(X))*gini_func(y_split_right_side)
    return giniProbability
print(gini(X,y,'island',0))

"""## 3.3
In order for the decision tree to work successfully, continuous variables need to be converted to categorical variables first. To do this, you need to implement a decision function that makes this split. Let us call that cont_to_cat(). The details of the function are the following. [10 marks]:-
a. Assume that the continuous variables are independent of each other   i.e. assuming 2 continuous variables A and B, the split of A does not in any way affect the split you will perform in B. 
b. The continuous variables should only be split into 2 categories, and the optimal split is one that divides the samples the best, based on the value of the function you have been allotted (as per your roll number).
"""

X['bill_length_mm'].dtype

def cont_to_cat(X, y):
    continous_data = ['bill_depth_mm', 'bill_length_mm', 'flipper_length_mm', 'body_mass_g']
    for column_name in continous_data:
        n = (max(X[column_name]) - min(X[column_name])) / 1500
        splt = np.arange(min(X[column_name]), max(X[column_name]), n)
        split_values = [gini(X, y, column_name, i) for i in splt]
        split_values_min_gini = splt[np.argmin(split_values)]
        X[column_name] = np.where(X[column_name] <= split_values_min_gini, 0, 1)
    return X
X = cont_to_cat(X, y)

"""## 3.4
After step 2, all the attributes would have categorical values, so now you can go ahead and implement the training function. This would include implementing the following helper functions: [25 marks]
a. Get the attribute that leads to the best split
"""

def bestSplit(X, y):
    least_gini_impurity = 1.0
    best_feature_name = None
    Split_Value = 0
    for i in X.columns:
      unique_values_in_feature = []
      for j in X[i]:
        if j not in unique_values_in_feature:
          unique_values_in_feature.append(j)
      for value in unique_values_in_feature:
        score = gini(X, y,i,value)
        if score < least_gini_impurity:
            least_gini_impurity = score
            Split_Value = value
            best_feature_name = i
            #print(value)
            #print(least_gini_impurity)
            #print(best_feature_name)
    return best_feature_name,Split_Value
best_feature_name,Split_Value = bestSplit(X, y)

"""b. Make that split """

Xleft_child = X[X[best_feature_name] <= Split_Value]
Xright_child = X[X[best_feature_name] > Split_Value]
yleft_child = y[X[best_feature_name] <= Split_Value]
yright_child = y[X[best_feature_name] > Split_Value]

left_child

right_child

"""c. Repeat these steps for the newly-created split"""

def check_child_Leaf_Node(y):
  if len(set(y)) <= 1:
    return True
  else: 
    return False

class node:
  def __init__(self, parent,feature,X,y, depth):
    self.feature = feature
    self.X = X
    self.y = y
    self.pVal = None
    self.children = []
    self.parent = parent
    self.depth = depth



class DecisionTree:
  def __init__(self, X, y, max_depth):
    self.X = X
    self.y = y
    self.max_depth = max_depth
  def gini_func(self, dataset):  
    unique_classes, class_counts = np.unique(dataset, return_counts=True)    
    n=len(dataset)
    prob = class_counts/n
    p_s=0
    for i in range(len(unique_classes)):
      p_s+=prob[i]**2
    gini_cost=1-p_s       
    return(gini_cost)
  def gini(self, X, y, feature_name, split_value):
    X_split_left_side = X[X[feature_name] <= split_value][feature_name]
    y_split_left_side = y[X[feature_name] <= split_value]
    X_split_right_side = X[X[feature_name] > split_value][feature_name]
    y_split_right_side = y[X[feature_name] > split_value]
    l1 = len(X_split_left_side)
    l2 = len(X_split_right_side)
    giniProbability = (l1/len(X))*self.gini_func(y_split_left_side) + (l2/len(X))*self.gini_func(y_split_right_side)
    return giniProbability

  def bestSplit(self, X, y):
    least_gini_impurity = 1.0
    best_feature_name = 'flipper_length_mm'
    Split_Value = 0
    for i in X.columns:
      unique_values_in_feature = []
      for j in X[i]:
        if j not in unique_values_in_feature:
          unique_values_in_feature.append(j)
      for value in unique_values_in_feature:
        score = gini(X, y,i,value)
        if score <= least_gini_impurity:
            least_gini_impurity = score
            Split_Value = value
            best_feature_name = i
            #print(value)
            #print(least_gini_impurity)
            #print(best_feature_name)
    return best_feature_name,Split_Value

  def leaf_node(self, node_):
    return self.gini_func(node_.y) == 0 or check_child_Leaf_Node(node_.y) or node_.depth > self.max_depth

  def dfs(self, node_):

    if (self.leaf_node(node_)):
      return node_

    node_.feature, splt_val = self.bestSplit(node_.X, node_.y)
    print(node_.feature)
    node_.pVal = splt_val
    Xleft_child = node_.X[node_.X[node_.feature] <= splt_val]
    Xright_child = node_.X[node_.X[node_.feature] > splt_val]
    yleft_child = node_.y[node_.X[node_.feature] <= splt_val]
    yright_child = node_.y[node_.X[node_.feature] > splt_val]

    node_.children += [self.dfs(node(node_, None, Xleft_child, yleft_child,node_.depth+1))]
    node_.children += [self.dfs(node(node_, None, Xright_child, yright_child,node_.depth+1))]

    return node_

  def train(self):
    self.root = self.dfs(node(None, None, self.X, self.y, 0))

  def classify(self, node_, x):
    if (self.leaf_node(node_)):
      return int(node_.y.mode()[0])
    
    if (x[node_.feature] <= node_.pVal):
      return self.classify(node_.children[0], x)
    else:
      return self.classify(node_.children[1], x)

  def test(self, X_test, y_test):
    correct_classifications = 0

    for i in range(len(X_test.index)):
      x = X_test.iloc[i]
      y = y_test.iloc[i]
      if (y == self.classify(self.root, x)):
        correct_classifications+=1

      return correct_classifications/len(X_test.index) *100.0

dct = DecisionTree(X_train, y_train, 10)
dct.train()

X_test.iloc[len(X_test.index) - 1]

dct.test(X_test, y_test) * 100.0
