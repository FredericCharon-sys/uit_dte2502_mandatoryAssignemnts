import numpy as np
import time

np.random.seed(2)

'''
For the activation function we replace relu() by the sigmoid() function, sigmoid_grad() is the derivative of sigmoid()

The difference between this functions is that the relu function starts to have more random error results in the 
    beginning before approaching the best weights, the sigmoid function gets more stable after less iterations. This is 
    because the relu function can return big positive values while the sigmoid function cannot return a value >1. Inputs 
    close to x=0 also have less impact on the network for the sigmoid function.
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    # return np.exp(-x) / (1 + np.exp(-x) ** 2)
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return (x > 0) * x

def relu_grad(x):
    return x > 0

streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])

walk_vs_stop = np.array([[0], [1], [0], [1], [1], [0]])
X, y = streetlights, walk_vs_stop

'''
We add a new hidden layer between layer1 and layer_out, for this new layer2 we define the number of nodes first
6 nodes seem to work pretty good, too many or too few nodes will make the result less accurate
'''
hidden_nodes_layer1 = 8
hidden_nodes_layer2 = 6

'''
A value vor epochs and learning rate is initialized. When using a high learning rate, the program will approach the
    optimal weights way faster and you need less epochs but we might overshoot the correct values. If the learning rate
    is too low we will approach slower and not overshoot but it will take many more epochs to get the right weights
    That's why you need to find ta good balance between these options.
'''
epochs = 100  # number of iterations to go through the network
lr = 0.1  # how much we change the weights of the network each iteration

'''
we're adding a 3rd weights matrix for our new layer. We connect ws_2 to the new hidden nodes and ws_3 to the output
'''
ws_1 = np.random.rand(X.shape[1], hidden_nodes_layer1) - 0.5
ws_2 = np.random.rand(hidden_nodes_layer1, hidden_nodes_layer2) - 0.5
ws_3 = np.random.rand(hidden_nodes_layer2, y.shape[1]) - 0.5

for epoch in range(epochs):  # number of training iterations, or times to change the weights of the nn
    for i in range(X.shape[0]):  # for all samples in X, each streetlight
        layer_in = X[i:i + 1]

        '''
        adding the new layer_2 in between layer_1 and layer_out and implementing ws_3
        '''
        # forward pass/prediction
        layer_1 = sigmoid(layer_in.dot(ws_1))
        layer_2 = sigmoid(layer_1.dot(ws_2))
        layer_out = layer_2.dot(ws_3)

        '''
        Since we added a new layer we also have to include a new delta value for the backpropagation
        '''
        # calc error/distance (how far are we from goal)
        delta_3 = layer_out - y[i:i + 1]

        # calc the the error each node in prev layer contributed
        delta_2 = delta_3.dot(ws_3.T) * sigmoid_grad(layer_2)
        delta_1 = delta_2.dot(ws_2.T) * sigmoid_grad(layer_1)

        '''
        Since we added a new weight matrix ws_3 it also has to be updated in the end
        '''
        # update weights
        ws_3 -= lr * (layer_2.T.reshape(hidden_nodes_layer2, 1).dot(delta_3))
        ws_2 -= lr * (layer_1.T.reshape(hidden_nodes_layer1, 1).dot(delta_2))
        ws_1 -= lr * (layer_in.T.reshape(X.shape[1], 1).dot(delta_1))


    if epoch % 10 == 0:
        '''
        instead of delta_2 we print out delta_3 since this is our new error value for the output layer
        '''
        error = delta_3 ** 2
        print(round(error[0][0], 6))  # , end='\r')
