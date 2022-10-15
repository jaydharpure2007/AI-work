'''
Gradient descent for simple linear regression model 
for single input
'''
import numpy as np
def gradient_descent(x,y):
    w1 = b1 = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.001

    for i in range(iterations):
        y_predicted = w1 * x + b1
        cost = (2/n) * sum([val**2 for val in (y-y_predicted)])
        w1d = -(1/n)*sum(x*(y-y_predicted))
        b1d = -(1/n)*sum(y-y_predicted)
        w1 = w1 - learning_rate * w1d
        b1 = b1 - learning_rate * b1d
        print ("w1 {}, b1 {}, cost {} iteration {}".format(w1,b1,cost, i))

x = np.array([1,2,3,4,5,6])
y = np.array([5,7,9,11,13, 15])

gradient_descent(x,y)

'''
Gradient descent for simple linear regression model 
using sigmoid activation function for single input
'''
def gradient_descent_sig(x,y):
    w1 = b1 = 0
    iterations = 5
    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        neth1 = w1 * x + b1
        outh1 = sigmoid(neth1)
        cost = (2/n) * sum([val**2 for val in (y-outh1)])
        w1d = -(1/n)*sum((y-outh1)*outh1*(1-outh1))
        b1d = -(1/n)*sum(y-outh1)
        w1 = w1 - learning_rate * w1d
        b1 = b1 - learning_rate * b1d
        print ("m {}, b {}, cost {} iteration {}".format(w1,b1,cost, i))

x = np.array([0.1,0.2,0.3,0.4,0.5])
y = np.array([0.1,0.6,0.9,0.4,0.98])

gradient_descent_sig(x,y)

def sigmoid (x):
    s = 1/(1+np.exp(-x))
    return s