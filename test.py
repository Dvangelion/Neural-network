import numpy as np
import pandas as pd

#define constants
#N is the number of test data you want to load
N = 500
#a is the starting column number in training data,notice a+N < 27999
a = 18500
#b is the ending column number in training data
b = a + N

#load model
model = np.load('model.npz')
W1,W2,b1,b2 = model['W1'],model['W2'],model['b1'],model['b2']

#load data
#train.csv contains data of hand written digits and targets(the actual digit)
train = pd.read_csv('train.csv')
dataframe = pd.DataFrame(train)
#first column in data is the target
target = dataframe.ix[:,0]

#test_train is a 784 by N array, stores N columns of data
test_train = np.zeros((784,N))

#stores ith row in train.csv to ith column in test_train
for i in range(N):
    test_train[:,i] = dataframe.ix[i+a][1:]

#extract appropriate range of target_test from target
target_test = np.array(target[a:b])


def convert_prediction(prediction):
    '''
    :param prediction: prediction is an array of size (10,N), stored probabilities of given data is 0~9 in
    each column. Please see lab report to find examples.

    :return: an array of size (1,N), each element is the digit of highest probability.

    '''
    #zero array of length N
    target = np.zeros(prediction.shape[1])

    for i in range(prediction.shape[1]):
        #fill the ith element in target with digit of highest probability
        target[i] = np.argmax(prediction[:,i])
    return target


def accuracy_test():
    #forward propagation
    h_input = np.dot(W1.T, test_train) + b1  # Input to hidden layer.
    h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
    logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
    prediction = 1 / (1 + np.exp(-logit))  # Output prediction.

    #convert prediction to vector, easier to compare with targets in training data
    prediction = convert_prediction(prediction)
    #calculate accuracy
    print 'the accuracy is',sum(prediction == target_test)/float(N)
    return 0

accuracy_test()
