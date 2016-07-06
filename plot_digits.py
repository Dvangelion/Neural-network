import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#load test data
#test.csv contains data of hand written digits and targets(the actual digit)
test = pd.read_csv('test.csv')
dataframe = pd.DataFrame(test)

#first column in train.csv is the target
target = dataframe.ix[:,0]

#load model
model = np.load('model.npz')
#W1,W2 are weight parameters, b1,b2 are bias terms.
W1,W2,b1,b2 = model['W1'],model['W2'],model['b1'],model['b2']


def convert_target(target):
    '''
    :param target:target is a vector of length N, each value is the actual digit.
    :return: returns a 10 by N array, filled by 0 and 1. In each column, 1 is the actual digit. And elsewhere are 0.
    Please see lab report for example.
    '''
    empty_target = np.zeros((len(target),10))
    for i in range(len(target)):
        empty_target[i,target[i]] = 1
    return empty_target

def convert_prediction(prediction):
    '''
    :param prediction: prediction is an array of size (10,N), stored probabilities of given data is 0~9 in
    each column. Please see lab report to find examples.

    :return: an array of size (1,N), each element is the digit of highest probability.

    '''

    #initialize target as a zero array of length N
    target = np.zeros(prediction.shape[1])

    for i in range(prediction.shape[1]):
        #fill the ith element in target with digit of highest probability
        target[i] = np.argmax(prediction[:,i])
    return target

def show_digit():
    '''
    Plot the digit, and prediction result
    '''

    digit_num = input('input the digit number you want to plot(0~27999)')
    if 0<= digit_num<= 27999:
        digit = dataframe.ix[digit_num]
    else:
        raise NameError('digit number must be 0~27999')

    #forward propagation
    h_input = np.array([np.dot(W1.T, digit)]).T + b1  # Input to hidden layer.
    h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
    logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
    prediction = 1 / (1 + np.exp(-logit))  # Output prediction.
    prediction = convert_prediction(prediction) #convert prediction from a vector to a number

    #reshape data for plot purpose
    digit = digit.reshape(28,28) # reshape digit to 28 by 28 array

    #make the plot
    plt.imshow(digit,cmap=plt.cm.gray) # plot digit
    plt.title('digit number'+' '+str(digit_num))
    plt.xlabel('prediction:'+str(prediction[0]))
    plt.show()
    show_digit()#recursion

show_digit()