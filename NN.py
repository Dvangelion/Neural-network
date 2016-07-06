import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def InitNN(num_inputs, num_hiddens, num_outputs):
  """
  Initializes NN parameters.
  W1,W2 are weight parameters, b1,b2 are bias terms.

  """
  W1 = np.random.randn(num_inputs, num_hiddens)
  W2 = np.random.randn(num_hiddens, num_outputs)
  b1 = np.zeros((num_hiddens, 1))
  b2 = np.zeros((num_outputs, 1))
  return W1, W2, b1, b2

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


def TrainNN(num_hiddens, eta, momentum, num_iters,N):

  #intialize training
  inputs_train = np.zeros((784,N))

  #load test data
  #train.csv contains data of hand written digits and targets(the actual digit)
  test = pd.read_csv('train.csv')
  dataframe = pd.DataFrame(test)
  #first column in data is the target
  target = dataframe.ix[:,0]

  #stores ith row in train.csv to ith column in test_train
  for i in range(N):
    inputs_train[:,i] = dataframe.ix[i][1:]

  #convert sigle value target to vector of legnth 10. Please see conver_target() for more comment.
  target_train = convert_target(np.array(target[:N])).T

  #Initialize W1,W2,b1,b2,and their derivatives
  W1, W2, b1, b2 = InitNN(inputs_train.shape[0], num_hiddens, target_train.shape[0])
  dW1 = np.zeros(W1.shape)
  dW2 = np.zeros(W2.shape)
  db1 = np.zeros(b1.shape)
  db2 = np.zeros(b2.shape)

  #Create empty list to store cross entropy
  train_error = []

  #Main loop
  for epoch in range(num_iters):

    # Forward pass
    h_input = np.dot(W1.T, inputs_train) + b1  # Input to hidden layer a_j.
    h_output = 1 / (1 + np.exp(-h_input))  # Output from hidden layer a_j to z_j.

    logit = np.dot(W2.T, h_output) + b2  # Input to output layer o_k.
    prediction = 1 / (1 + np.exp(-logit))  # Output prediction y_k.

    # Compute cross entropy
    train_CE = -np.mean(target_train * np.log(prediction) + (1 - target_train) * np.log(1 - prediction))

    # Compute derivative
    dEbydlogit = prediction - target_train # dE by d(o_k) = (o_k - t_k)

    # Backpropagation
    dEbydh_output = np.dot(W2, dEbydlogit) # dE by d(z_j) = W2(o_k - t_k)
    dEbydh_input = dEbydh_output * h_output * (1 - h_output) # dE by d(a_j) = dE by d(z_j)*z_j*(1-z_j)

    # Gradients for weights and biases.
    dEbydW2 = np.dot(h_output, dEbydlogit.T) #   dEbydW2 = (o_k - t_k)z_j
    dEbydb2 = np.sum(dEbydlogit, axis=1).reshape(-1, 1) #   dEbydb2 = sum of dE by d(o_k)
    dEbydW1 = np.dot(inputs_train, dEbydh_input.T) #   dEbydW1 = dE by d(a_j)*x_i
    dEbydb1 = np.sum(dEbydh_input, axis=1).reshape(-1, 1) #   dEbydb1 = sum of dE by d(a_j)

    # Update the weights at the end of the iteration
    dW1 = momentum * dW1 - (eta / N) * dEbydW1
    dW2 = momentum * dW2 - (eta / N) * dEbydW2
    db1 = momentum * db1 - (eta / N) * dEbydb1
    db2 = momentum * db2 - (eta / N) * dEbydb2

    W1 = W1 + dW1
    W2 = W2 + dW2
    b1 = b1 + db1
    b2 = b2 + db2

    #store cross entropy
    train_error.append(train_CE)


  return W1, W2, b1, b2, train_error


def DisplayErrorPlot(train_error):
  '''

  :param train_error: train error is the cross entropy in each iteration.
  This function is used to make a plot of cross entropy versus iteration steps.
  '''

  plt.figure(1)
  plt.clf()
  plt.plot(range(len(train_error)), train_error, 'b')
  plt.xlabel('Iters')
  plt.ylabel('Cross entropy')
  plt.title('Cross entropy versus Iters')
  plt.show()
  raw_input('Press Enter to exit.')

def SaveModel(modelfile, W1, W2, b1, b2, train_error):
  """Saves the model to a numpy file."""
  model = {'W1': W1, 'W2' : W2, 'b1' : b1, 'b2' : b2,
           'train_error' : train_error}
  print 'Writing model to %s' % modelfile
  np.savez(modelfile, **model)

def main():
  #parameters, please see lab report for explanation
  N = 3500
  num_hiddens = 100
  eta = 0.5
  momentum = 0.5
  num_iters = 3500

  W1, W2, b1, b2, train_error = TrainNN(num_hiddens, eta, momentum, num_iters, N)
  DisplayErrorPlot(train_error)
  #save the model for future use:
  outputfile = 'model.npz'
  #SaveModel(outputfile, W1, W2, b1, b2, train_error)




if __name__ == '__main__':
  main()




