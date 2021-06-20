import numpy as np


class Sigmoid:
    def forward(self, coming_matrix):
        x = np.where(coming_matrix < 700, coming_matrix, 700)
        x = np.where(x > -700, x, -700)
        self.output = 1/(1+np.exp(-x))
        self.backward(self.output)

    def backward(self, coming_matrix):
        self.back = coming_matrix * (1-coming_matrix)

    def delta(self, error):
        return error*self.back


class Linear:
    def forward(self, coming_matrix):
        self.output = coming_matrix
        self.backward(self.output)

    def backward(self, coming_matrix):
        self.back = np.ones_like(coming_matrix)

    def delta(self, error):
        return error*self.back


class Relu:
    def forward(self, coming_matrix):
        self.output = np.maximum(0, coming_matrix)
        self.backward()

    def backward(self):
        self.back = np.greater(self.output, 0).astype(int)

    def delta(self, error):
        return error*self.back


class Soft_Max:
    def forward(self, coming_matrix):
        self.output = np.exp(coming_matrix) / np.sum(np.exp(coming_matrix), axis=1).reshape(-1,1)
        s = np.copy(self.output)

        identity = np.eye(s.shape[-1])
        df = s[...,np.newaxis] * identity

        z = np.zeros_like(df)
        for i in range(len(coming_matrix)):
            z[i] = np.dot(s[i][...,np.newaxis], s[i][...,np.newaxis].T)
            
        jacob = df - z
        self.back = jacob

    def delta(self, error):
        error = error.reshape((error.shape[0], error.shape[1], 1))
        delta = np.zeros_like(error)
        for _ in range(delta.shape[0]):
            delta[_] = np.dot(self.back[_], error[_])
        delta = delta.reshape(delta.shape[0], delta.shape[1])
        return delta
