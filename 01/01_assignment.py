import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
Setting up the dataset with pandas as shown for the assignment.
We're going to classify between the iris plant types Iris-setosa and Iris-versicolor by using the attributes 
    petal length and petal width 
'''
dataset = pd.read_csv('iris.csv', header=None, names=['sepal length', 'sepal width', 'petal length',
                                                      'petal width', 'class'])
dataset.drop('sepal length', axis='columns', inplace=True)
dataset.drop('sepal width', axis='columns', inplace=True)
binary_dataset = dataset.drop(index=dataset.index[dataset['class'] == 'Iris-virginica'])

binary_dataset.loc[dataset['class'] == 'Iris-setosa', dataset.columns == 'class'] = 0
binary_dataset.loc[dataset['class'] == 'Iris-versicolor', dataset.columns == 'class'] = 1

'''
implementing the sigma() and the training() function for the OR operator as shown in the lecture
'''
# neuron activation
def sigma(x, w):
    activation = w[0]
    for i in range(len(x) - 1):
        activation += w[i + 1] * x[i]
    return 1.0 if activation > 0.0 else 0.0


# training function
def training(data, w0, mu, T):
    w = w0
    for t in range(T):
        for x in data:
            activation = sigma(x, w)
            error = x[-1] - activation
            w[0] = w[0] + mu * error
            for i in range(len(x) - 1):
                w[i + 1] = w[i + 1] + mu * error * x[i]
    return w


'''
Transforming the pandas binary_dataset to a matrix/nested_list so the training() function works as expected
'''
data = []
for idx, row in binary_dataset.iterrows():
    data.append([row['petal length'], row['petal width'], row['class']])
print("This is the data matrix: ", data)

'''
Initializing the weights
'''
weights = [0.3, 0.5, -0.1]

'''
The training() function is called using the data list we created
'''
weights = training(data, weights, 0.37, 10)
print("final weights: ", weights)

'''
All the data from the dataset is being plotted and colored depending on the classifier's output
'''
for d in data:
    if sigma(d, weights) == 1:
        plt.plot(d[0], d[1], 'ro')
    else:
        plt.plot(d[0], d[1], 'bo')

'''
We draw the classifier's function that separates the two classes
'''
x1 = np.linspace(0, 6, 100)
x2 = (weights[1] * x1 + weights[0]) / (-weights[2])
plt.plot(x1, x2, '-g')

'''
Creating the graph
'''
plt.axis([0, 6, 0, 2])
plt.grid()
plt.show()
