import pandas as pd
from matplotlib import pyplot
ds = pd.read_csv('MNIST_dataset.csv')
labels = ds.pop('labels')
x = ds.values.reshape(-1, 28, 28)
for i in range(10):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()