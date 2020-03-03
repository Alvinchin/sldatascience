from sklearn import tree
from sklearn import neural_network
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import numpy as np

clf_tree = tree.DecisionTreeClassifier()
clf_neural = neural_network.MLPClassifier()
clf_KNN = neighbors.KNeighborsClassifier()
clf_svm = svm.SVC()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf_neural = clf_neural.fit(X, Y)
clf_tree = clf_tree.fit(X, Y)
clf_KNN = clf_KNN.fit(X,Y)
clf_svm = clf_svm.fit(X,Y)



neural_prediction = clf_neural.predict(X)
acc_neural = accuracy_score(Y, neural_prediction) * 100
print('Accuracy for NeuralNetwork: {}'.format(acc_neural))

tree_prediction = clf_tree.predict(X)
acc_tree = accuracy_score(Y, tree_prediction) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

knn_predication = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, knn_predication) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

svm_predication = clf_svm.predict(X)
acc_svm = accuracy_score(Y, svm_predication) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

index = np.argmax([acc_neural,acc_svm, acc_KNN])
classifiers = {0: 'NeuralNetwork', 1: 'SVM', 2: 'KNN'}
print('Best gender classifier is {}'.format(classifiers[index]))

