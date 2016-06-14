import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import scipy.optimize as op

def sigmoid(arr):
    return 1/(1 + np.exp(-1 * arr))

def gradient(theta,X, y):
    grad = np.zeros(theta.shape)
    skipflag = 0
    hx = sigmoid(np.dot(X , theta))
    for i in range(theta.shape[0]):
        grad[i] = np.sum(np.multiply(np.subtract(hx, y), X[0::, i])) / y.shape[0]
    return grad

def normalizeArray(arr):
    #First fill all the empty values by the average of non-empty values
    notNullValues = arr[arr != ''].astype(np.float)
    averageNotNull = np.mean(notNullValues)
    arr[arr == ''] = averageNotNull
    arr = arr.astype(np.float)
    finalAverage = np.mean(arr)
    stdDev = np.std(arr)
    arr = (np.absolute(np.subtract(arr,finalAverage)))/stdDev
    return arr
    
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

plt.plot(survivedIndices[0::,5], survivedIndices[0::,9], 'x')
plt.plot(notsurvivedIndices[0::, 5], notsurvivedIndices[0::,9], 'ro')
plt.axis([np.amin(age), np.amax(age) , np.amin(fare) , np.amax(fare) ])
plt.show()

ones = np.ones((np.size(train_data[0::,0]), 1))
X = np.column_stack((ones, train_data[0::, 5], train_data[0::, 9]))
y = train_data[0::, 1].astype(np.float)
initialTheta = np.ones((X.shape[1], 1))

X = np.asmatrix(X)
y = np.asmatrix(y).transpose()

initCost = costFunction(initialTheta, X, y)
initGrad = gradient(initialTheta, X, y)

print "Initial Cost: " + str(initCost)
print "Initial Gradient: " + str(initGrad)

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

age = test_data[0::, 4]
fare = test_data[0::,8]

age = normalizeArray(age)
fare = normalizeArray(fare)
ones = np.ones(np.size(test_data[0::,0]))
X = np.column_stack((ones, age, fare))

optimalTheta = np.transpose(optimalTheta)
hx = sigmoid(np.dot(X , optimalTheta))

hx[hx >= 0.5] = 1
hx[hx < 0.5] = 0

resultList = hx[0::, 0].tolist()

for i in range(len(resultList)):      
    writer_obj.writerow([test_data[i][0], "%d" %int(resultList[i][0])])

prediction_file.close()
