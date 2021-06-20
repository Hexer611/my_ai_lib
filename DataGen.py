import numpy as np
from math import cos, sin
import pandas

np.random.seed(0)


class DataGeneration:
    def __init__(self, func=''):
        if func == 'cos':
            self.cos_function_dataset()
        elif func == 'sin':
            self.sin_function_dataset()
        elif func == 'and':
            self.and_operation_dataset()
        elif func == 'spiral':
            self.spiral_data()
        elif func == 'numbers':
            self.minst_numbers_traindata()

    def and_operation_dataset(self):
        self.data = np.array([[0,1], [1,0], [0,0], [1,1]]).astype('uint8')
        self.exp_out = np.array([[0], [0], [0], [1]])

    def cos_function_dataset(self):
        # astype makes both self.data and self.exp_out float16
        self.data = np.array(np.arange(0, 6.4, 0.5)).reshape(-1, 1).astype('float16')
        self.exp_out = np.cos(self.data)
        self.data /= max(self.data)

    def sin_function_dataset(self):
        # astype makes both self.data and self.exp_out float16
        self.data = np.array(np.arange(0, 6.4, 0.5)).reshape(-1, 1).astype('float16')
        self.exp_out = np.sin(self.data)
        self.data /= 10

    def spiral_data(self, points=100, classes=2): #https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c
        X = np.zeros((points*classes, 2))
        y = np.zeros((points*classes, classes), dtype='uint8')
        for class_number in range(classes):
            ix = range(points*class_number, points*(class_number+1))
            r = np.linspace(0.0, 1.0, points)  # radius
            t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
            X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
            y[ix, class_number] = 1
        self.data = X
        self.exp_out = y

    def minst_numbers_traindata(self, loadfile='Datasets/Mnist-Numbers/mnist_train.mynp'):
        #loaded_data = pandas.read_csv(d_path, sep=',', header=None, dtype='float16').to_numpy()

        with open(loadfile, 'rb') as f:
            loaded_data = np.load(f)

        print("omen")
        X = np.asarray(loaded_data[:, 1:]) / 255 + 0.01
        y = np.zeros((X.shape[0], 10), dtype='float16')
        for _ in range(X.shape[0]):
            y[_,int(loaded_data[_,0])-1] = 1
        #y = np.asarray(loaded_data[:, 0]) / 10.0
        self.puredata = loaded_data[:, 1:].astype('uint8')
        self.data = X
        self.exp_out = y


def csv_to_numpybinary(loadfile='Datasets/Mnist-Numbers/mnist_test.csv', savefile='Datasets/Mnist-Numbers/mnist_test.mynp'):
    loaded_csv = pandas.read_csv(loadfile, sep=',', header=None, dtype='float16').to_numpy()
    with open(savefile, 'wb') as f:
        np.save(f, loaded_csv)

#csv_to_numpybinary()
