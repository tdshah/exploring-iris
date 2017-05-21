#This project explores ML concepts using the popular iris dataset
#small end-to-end project that loads the data and trains 6 different models/evaluates different algorithms to ultimately make the best predictions
#This file can be used as a template for visualization & training utilizing other datasets

import sys
import scipy
import numpy
import matplotlib

import sklearn


import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#this line prints out the number of instances (Rows) and attributes (columns) the data contains
print(dataset.shape)

#before moving further, peek the data so you know everything is in order
#use head
print(dataset.head(50))

#summary of data
print(dataset.describe())

print(dataset.groupby('class').size())

#################################################################################################################################3
#data visualizations- univariate. Understand individual attributes better

#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#histograms
dataset.hist()
plt.show()

#data visualizations- multivariate. Understand the relationship between attributes

#scatterplots. these help detect structured relationships through displaying correlations
scatter_matrix(dataset)
plt.show()


#create cross-validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

seed = 7
scoring = 'accuracy'


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

#compare the 6 models (algorithms) used by visualizing them and looking at their mean accuracy for the 10 iterations performed in 10 fold cross validation
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



#test the best model (KNN, since it had the highest accuracy) on the validation set

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#this concludes that KNN will provide the best predictions (i.e. highest accuracy) for the kind of flower 