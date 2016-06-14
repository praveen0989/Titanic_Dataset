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

'''
Computes the sigmoid function value of the given parameter
The parameter arr can be a Scalar or a Vector(matrix)
Returns the data type corresponding to the passed parameter.
'''
def sigmoid(arr):
    return 1/(1 + np.exp(-1 * arr))

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

'''
Normalises and array of values passed. First, all the empty values in the array are filled up
by computing the mean of non-empty values. Them value = (value - mean)/ standard deviation transformation
is applied to all the values. Normalisation is done so that different feature vectors have similar values.
'''
def normalizeArray(arr):
    notNullValues = arr[arr != ''].astype(np.float)
    averageNotNull = np.mean(notNullValues)
    arr[arr == ''] = averageNotNull
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
    hx = sigmoid(np.dot(X , theta))
    oneMinusY = np.ones(y.shape) - y
    oneMinusHx = np.ones(hx.shape) - hx
    yloghx = np.multiply(y, np.log(hx))
    oneMinusloghx = np.multiply(oneMinusY, np.log(oneMinusHx))
    sumOfError = np.add(yloghx, oneMinusloghx)
    cost = np.sum(sumOfError)* -1 / y.shape[0]
    return cost


train_file = csv.reader(open('train.csv', 'rb'))
header = train_file.next()

train_data = []
for row in train_file:
    train_data.append(row)

train_data = np.array(train_data)
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
X = np.column_stack((ones.astype(np.float), train_data[0::, 5].astype(np.float), train_data[0::, 9].astype(np.float)))
y = train_data[0::, 1].astype(np.float)

#Assume an initial value of theta.
initialTheta = np.ones((X.shape[1], 1))

X = np.asmatrix(X)
y = np.asmatrix(y).transpose()

initCost = costFunction(initialTheta, X, y)
initGrad = gradient(initialTheta, X, y)

print "Initial Cost: " + str(initCost)
print "Initial Gradient: " + str(initGrad)

'''
Scipy optimise module has a method to minimise the cost of the model by fitting theta
appropriately. The method parameter can be varied to run a differnt algorithm to compute
optimal value of theta.
'''
print "Computing Optimal theta..."

result = op.minimize(fun = costFunction, x0 = initialTheta, args = (X, y), method = 'TNC', jac = gradient)
optimalTheta = np.asmatrix(result.x)

print "Optimal Theta: " + str(optimalTheta)

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

age = normalizeArray(age)
fare = normalizeArray(fare)
ones = np.ones(np.size(test_data[0::,0]))
X = np.column_stack((ones, age, fare))

optimalTheta = np.transpose(optimalTheta)

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
