import numpy as np
import pandas as pd

N = 28000
test = pd.read_csv('test.csv')
dataframe = pd.DataFrame(test)

model = np.load('model.npz')
W1,W2,b1,b2 = model['W1'],model['W2'],model['b1'],model['b2']
test_train = np.zeros((784,N))

for i in range(N):
    test_train[:,i] = dataframe.ix[i]
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

h_input = np.dot(W1.T, test_train) + b1  # Input to hidden layer.
h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
prediction = 1 / (1 + np.exp(-logit))  # Output prediction.

#convert prediction to vector, easier to compare with targets in training data
prediction = convert_prediction(prediction).astype(int)
result = pd.DataFrame(prediction)
result.to_csv('result.csv',index=False)