import matplotlib.pyplot as plt
import numpy as np


class Visualize:
    def __init__(self, row=None, column=None, xplt=(-1, 1), yplt=(-1, 1), t_layout=None, fgsize=None, xtcks=None, ytcks=None):
        self.xplt = xplt
        self.yplt = yplt
        self.xtcks = xtcks
        self.ytcks = ytcks
        if row is None:
            self.fig, self.fs = plt.subplots(1, 1, clear=True, figsize=(6,5))
        else:
            self.row = row
            self.column = column
            if fgsize is None:
                fgsize = (self.column*3,self.row*3)
            self.fig, self.fs = plt.subplots(row, column, clear=True, figsize=fgsize)
            self.fig.canvas.manager.window.wm_geometry('+100+0')  # Setting position of the plot window
        if t_layout is not None:
            self.fig.tight_layout(pad=t_layout)

    def function_2d_show(self, x, y, p):  # X=Input, Y=Expected output, P=Predicted output
        self.fs.plot(x, y)
        self.fs.plot(x, p)
        plt.pause(0.000001)
        self.fs.cla()

    def spiral_funcion_allinone_show(self, x, y, p): # X=Input, Y=Expected output, P=Predicted output
        for _ in range(len(p[0])):
            self.fs.scatter(x[:, 0][p[:, _] > 0.5], x[:, 1][p[:, _] > 0.5], cmap='winter_r')
        plt.pause(0.00001)  
        self.fs.cla()

    def spiral_funcion_show(self, x, y, p): # X=Input, Y=Expected output, P=Predicted output
        for rw in range(self.row):
            for cl in range(self.column):
                # Show every category seperately with expected value
                p_filter = p[:, cl] == np.amax(p, axis=1)
                self.fs[0, cl].scatter(x[:, 0][p_filter], x[:, 1][p_filter], cmap='winter_r')
                y_filter = y[:, cl] == np.amax(y, axis=1)
                self.fs[1, cl].scatter(x[:, 0][y_filter], x[:, 1][y_filter], cmap='winter_r')
                # Show every category seperately
                #self.fs[cs, category].scatter(X[:,0], X[:,1], c=P[:,category*self.column+cs], cmap='winter_r')
        plt.setp(self.fs, xlim=self.xplt, ylim=self.yplt)
        plt.pause(0.00001)
        for rw in range(self.row):
            for cl in range(self.column):
                self.fs[rw, cl].cla()

    def minst_numbers_show(self, x, y, p, offset=0, delay=0.00001):
        for rw in range(self.row//2):
            for cl in range(self.column):
                img = x[rw*self.column+cl+offset].reshape((28,28))
                self.fs[rw*2, cl].imshow(img, cmap='Greys')
                #self.fs[rw*2+1, cl].text(0,0, 'HELU')
                self.fs[rw*2+1, cl].bar([1,2,3,4,5,6,7,8,9,0], p[rw*self.column+cl+offset])
                plt.sca(self.fs[rw*2+1, cl])
                plt.xticks(self.xtcks, fontsize=8)
        plt.pause(delay)
        for rw in range(self.row):
            for cl in range(self.column):
                self.fs[rw, cl].cla()
