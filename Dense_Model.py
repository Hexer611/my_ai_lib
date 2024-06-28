from ErrorFunc import MeanSquaredError
from Activations import Linear
from my_helpers import mini_batch
import Layers
import numpy as np


class Model:
    trainDisplayFreq = 10
    
    def __init__(self, LayerS, load_file=''):
        self.LS = LayerS[:,0]                         # Layers
        self.AS = np.hstack((Linear(), LayerS[:, 1])) # Activations

        self.tot_layers = len(self.LS)

        self.ERR = MeanSquaredError()
        self.ERR.outSize = self.LS[-1].outputSize
        if load_file != '':
            self.LOAD(load_file)

    def forward(self, x):
        self.AS[0].forward(x)

        for _ in range(self.tot_layers):
            self.LS[_].f_prop(self.AS[_].output)
            self.AS[_+1].forward(self.LS[_].output)
        
        self.output = self.AS[-1].output

    def backward(self, x, y, lr):
        self.ERR.calc(self.output, y)
        DELTAS = []

        realerror = self.ERR.back*lr

        DELTAS.append(self.AS[-1].delta(realerror))

        for _ in range(1, self.tot_layers):
            local_error = np.dot(DELTAS[-1], self.LS[-_].weights.T)
            DELTAS.append(self.AS[-_-1].delta(local_error))

        DELTAS.reverse()

        for _ in range(self.tot_layers):
            self.LS[_].weights += np.dot(self.AS[_].output.T, DELTAS[_])
            self.LS[_].biases += np.sum(DELTAS[_], axis=0)

    def train(self, x, y, iterations=1000, lr=0.01, save_file='', batch_size=16):
        for _ in range(iterations):
            tot_error = 0
            tot_cases = len(y)
            true_cases = 0
            for bX, by in zip(mini_batch(x, batch_size), mini_batch(y, batch_size)):
                self.forward(bX)
                self.backward(bX,by,lr)
                tot_error += self.ERR.output
                for _p, _e in zip(self.output, by):
                    if _e[np.where(np.max(_p) == _p)].any() == 1.0:
                        true_cases += 1

            if _ % self.trainDisplayFreq == 0:
                if save_file != '':
                    self.SAVE(save_file)
                    # print('Saved..', end='\r')
                #print('Error : {0} - Accuracy: {1}'.format(tot_error, true_cases/tot_cases), end='\r')
                print('Error : {0} - Accuracy: {1}'.format(tot_error, true_cases/tot_cases))

    def print_weights(self):
        for _ in range(self.tot_layers):
            print(self.LS[_].weights)

    def print_biases(self):
        for _ in range(self.tot_layers):
            print(self.LS[_].biases)
    
    def SAVE(self, file_path):
        with open(file_path, 'wb') as f:
            for _ in range(self.tot_layers):
                np.save(f, self.LS[_].weights)
                np.save(f, self.LS[_].biases)

    def LOAD(self, file_path):
        with open(file_path, 'rb') as f:
            for _ in range(self.tot_layers):
                self.LS[_].weights = np.load(f)
                self.LS[_].biases = np.load(f)