'''
The objective of this script is to train a classification model using logistic regression.
The dataset is available on Kaggle. https://www.kaggle.com/c/titanic
2 important parameters namely age and fare are considered for classification.
Finally, the classifier is run on the test data and the results are writtern to an output file.

Training data --> train.csv
Test data --> test.csv
Output --> test_results_logistic.csv
'''
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy.special import expit

#Global Constants

LAMBDA = 0.1
MAX_POLY_ORDER = 10

'''
Computes the sigmoid function value of the given parameter
The parameter arr can be a Scalar or a Vector(matrix)
Returns the data type corresponding to the passed parameter.
'''
def sigmoid(arr):
    return expit(arr)

'''
Computes the gradient corresponding to each of theta values.
X is a matrix containing feature values
y is the actual result.
'''
def gradient(theta,X, y):
    grad = np.zeros(theta.shape)
    skipflag = 0
    hx = sigmoid(np.dot(X , theta))
    for i in range(theta.shape[0]):
        grad[i] = np.sum(np.multiply(np.subtract(hx, y), X[0::, i])) / y.shape[0]
    return grad


def fillMissingValues(arr):
    notNullValues = arr[arr != ''].astype(np.float)
    averageNotNull = np.mean(notNullValues)
    arr[arr == ''] = averageNotNull
    return arr

'''
Normalises and array of values passed. First, all the empty values in the array are filled up
by computing the mean of non-empty values. Them value = (value - mean)/ standard deviation transformation
is applied to all the values. Normalisation is done so that different feature vectors have similar values.
'''
def normalizeArray(arr):
    arr = arr.astype(np.float)
    finalAverage = np.mean(arr)
    stdDev = np.std(arr)
    arr = (np.absolute(np.subtract(arr,finalAverage)))/stdDev
    return arr

'''
Computes the cost function for a given feature matrix and results.
Cost = -1/m * sum(y * log(hx) + (1 - y) * log(1- hx))
'''
def costFunction(theta, X, y):
    theta = np.asmatrix(theta)
    theta = np.transpose(theta)
    hx = sigmoid(np.dot(X , theta))
    oneMinusY = np.ones(y.shape) - y
    oneMinusHx = np.ones(hx.shape) - hx
    yloghx = np.multiply(y, np.log(hx))
    oneMinusloghx = np.multiply(oneMinusY, np.log(oneMinusHx))
    sumOfError = np.add(yloghx, oneMinusloghx)
    cost = np.sum(sumOfError)* -1 / y.shape[0]
    return cost


def addPolynomialFeatures(matrix, column):
    
    columnVector = matrix[0::, column]
    for i in range(2, MAX_POLY_ORDER):
        computedColumn = np.power(matrix[0::,column].astype(np.float), i)
        computedColumn = normalizeArray(computedColumn)
        matrix = np.column_stack((matrix, computedColumn))
    return matrix

train_file = csv.reader(open('train.csv', 'rb'))
header = train_file.next()

train_data = []
for row in train_file:
    train_data.append(row)

train_data = np.array(train_data)

originalColumnSize = np.shape(train_data)[1]

train_data[0::, 5] = fillMissingValues(train_data[0::, 5])
train_data[0::, 9] = fillMissingValues(train_data[0::, 9])

train_data = addPolynomialFeatures(train_data, 5)
train_data = addPolynomialFeatures(train_data, 9)

train_data[0::, 5] = normalizeArray(train_data[0::, 5])
train_data[0::, 9] = normalizeArray(train_data[0::, 9])

survivedIndices = train_data[train_data[0::,1] == '1']
notsurvivedIndices = train_data[train_data[0::,1]=='0']

'''
Plot the positive and negative examples on a chart.
Visualising the data helps in understanding the kind of hypothesis \
function to be considered for modeling.
'''
plt.plot(survivedIndices[0::,5], survivedIndices[0::,9], 'x')
plt.plot(notsurvivedIndices[0::, 5], notsurvivedIndices[0::,9], 'ro')
plt.show()

# Append a column of 1's to the feature matrix. This column corresponds to Theta0 parameter
ones = np.ones((np.size(train_data[0::,0]), 1))
X = np.column_stack((ones.astype(np.float), train_data[0::, 5].astype(np.float), train_data[0::,9].astype(np.float)))

for j in range(originalColumnSize, np.shape(train_data)[1]):
    X = np.column_stack((X, train_data[0::,j].astype(np.float)))
    
y = train_data[0::, 1].astype(np.float)

X = np.asmatrix(X)
y = np.asmatrix(y).transpose()

# Assume a training set of 70%.
# We will verify the accuracy on the cross validation set(remaining 30%)

nRows = X.shape[0]
trainSetSize = np.floor(0.70 * nRows)
cvSetSize = nRows - trainSetSize

XTrain = X[0:trainSetSize, :]
yTrain = y[0:trainSetSize, :]

#Assume an initial value of theta.
initialTheta = np.asmatrix((np.zeros((XTrain.shape[1], 1))).astype(np.float))

'''
Scipy optimise module has a method to minimise the cost of the model by fitting theta
appropriately. The method parameter can be varied to run a differnt algorithm to compute
optimal value of theta.
'''
print "Computing Optimal theta..."
print "Cost after each iteration:"
result = op.minimize(fun = costFunction, x0 = initialTheta, args = (XTrain, yTrain), method = 'TNC', jac = gradient)
optimalTheta = np.asmatrix(result.x)
print "Optimal Theta: " + str(optimalTheta)

print "Running predictions on cross validation set..."

optimalTheta = np.transpose(optimalTheta)
hx = sigmoid(np.dot(XTrain , optimalTheta))
hx[hx >= 0.5] = 1
hx[hx < 0.5] = 0

result = (hx.astype(np.float) == yTrain.astype(np.float))
accuracy = (result[result == True].shape[1]) / float(result.shape[0])
print accuracy



test_file = csv.reader(open('test.csv', 'rb'))
header = test_file.next()
prediction_file = open('test_results_logistic.csv','wb')
writer_obj = csv.writer(prediction_file)

writer_obj.writerow(["PassengerId", "Survived"])

test_data = []
for row in test_file:
    test_data.append(row)

test_data = np.array(test_data)

age = test_data[0::,4]
fare = test_data[0::,8]
age = fillMissingValues(age)
fare = fillMissingValues(fare)
age = normalizeArray(age)
fare = normalizeArray(fare)
ones = np.ones(np.size(test_data[0::,0]))
X = np.column_stack((ones, age, fare))
X = addPolynomialFeatures(X, 1)
X = addPolynomialFeatures(X, 2)

'''
Use the optimal theta values and the model built to estimate
the values for test data.
'''
hx = sigmoid(np.dot(X , optimalTheta))

hx[hx >= 0.5] = 1
hx[hx < 0.5] = 0

resultList = hx[0::, 0].tolist()

for i in range(len(resultList)):      
    writer_obj.writerow([test_data[i][0], "%d" %int(resultList[i][0])])

prediction_file.close()
