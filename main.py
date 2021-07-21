import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from helper import *

data = pd.read_csv("test.csv")
data = data[['Fancy Words', 'Distance']]

# print(data.head())
# print(data.shape)
# print(data.info())
# print(data.describe())

matrix = np.array(data.values, "float")

X = matrix[:, 0]
y = matrix[:, 1]

X = X / (np.max(X))

"""
plt.plot(X, y, "b*")
plt.ylabel("Distance Throws")
plt.xlabel('Fancy Words Used')
plt.title('Fancy Words Vs. Distance Throws')
plt.show()
"""

m = np.size(y)
X = X.reshape([len(X), 1])
x = np.hstack([np.ones_like(X), X])

theta = np.zeros([2, 1])

# print(computeCost(x, y, theta, m))  # initial cost = 915.4979503125666

theta, J = gradient(x, y, theta, m)

# print(theta)  # optimal theta = [[1.64754116] [0.18855458]]
# print(J)

plt.plot(X, y, 'b*')
plt.plot(X, x @ theta, '-')
plt.ylabel("Distance Throws")
plt.xlabel('Fancy Words Used')
plt.title('Fancy Words Vs. Distance Throws')
plt.show()  # Shows Line Of Best Fit
print("Final Cost: " + str(computeCost(x, y, theta, m)))  # final cost = 756.4073367032673

"""
predict1 = [1,(164/np.max(matrix[:,0]))] @ theta
print(predict1) # prediction = [1.95677068]
"""
