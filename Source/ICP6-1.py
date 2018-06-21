from sklearn import datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB # Importing class from naive bayes model
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score


iris = datasets.load_iris() # Importing iris data set

x = iris.data # Storing feature matrix
y = iris.target # Storing response(target) vector


model = MultinomialNB() # Creating object
model.fit(x, y) # Fitting the data set into the model
scores = cross_val_score (model,x,y,cv =10,scoring="accuracy") # Calculating accuracy scores by creating testing and learning parts
""" 
Finding the accuracy score using the cross validation score, the data is split into equal 10 folds, from which one will be testing set and the other 9 will be
training sets. Ten scores will be found since we are iterating 10 times by changing the training and testing set.
"""
a = scores.mean()* 100
print ("The accuracy of each itteration is: {}".format (scores))
print ("The mean accuracy is : {} %".format (a))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
y_pred = model.predict(x_test)
print("Using another model of training size 80%")
print(metrics.accuracy_score(y_test, y_pred))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
y_pred = model.predict(x_test)
print("Using another model of training size 70%")
print(metrics.accuracy_score(y_test, y_pred))

