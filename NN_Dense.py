import numpy as np
from math import cos, sin
from Dense_Model import Model
from Layers import Dense_Layer
from Activations import Linear, Soft_Max, Relu, Sigmoid
from Visualizations import Visualize
from PIL import Image
import cv2 as cv

import time
import math
from DataGen import DataGeneration

from ErrorFunc import MeanSquaredError

np.random.seed(0)

# PARAMETERS
dataset = 'newnumbers'
train = True
predict = True
visualize = not True
load_file = ''
save_file = './denemewrite.brain'

if dataset == 'sin' or dataset == 'cos':
    my_dataset = DataGeneration(dataset)
    my_network = Model(np.array(((Dense_Layer(1,16),Relu()),
                                (Dense_Layer(16,16),Relu()),
                                (Dense_Layer(16,16),Relu()),
                                (Dense_Layer(16,1),Linear()))),
                                load_file)
    if visualize:
        my_plot = Visualize()
    for __ in range(400):
        my_network.train(my_dataset.data, my_dataset.exp_out, iterations=100, lr=0.002, save_file=save_file, batch_size=16)
        if visualize:
            my_network.forward(my_dataset.data)
            my_plot.function_2d_show(my_dataset.data, my_dataset.exp_out, my_network.output)
elif dataset == 'spiral':
    my_dataset = DataGeneration()
    my_dataset.spiral_data(100, 3)
    my_network = Model(np.array(((Dense_Layer(2, 8), Relu()),
                                (Dense_Layer(8, 16), Relu()),
                                (Dense_Layer(16, 32), Relu()),
                                (Dense_Layer(32, 32), Relu()),
                                (Dense_Layer(32, 16), Relu()),
                                (Dense_Layer(16, 8), Relu()),
                                (Dense_Layer(8, 3), Soft_Max()))),
                                load_file)
    if visualize:
        #my_plot = Visualize()
        my_plot = Visualize(2, 3)
    for __ in range(100):
        my_network.train(my_dataset.data, my_dataset.exp_out, iterations=10, lr=0.001, save_file=save_file, batch_size=16)
        if visualize:
            my_network.forward(my_dataset.data)
            #my_plot.spiral_funcion_allinone_show(my_dataset.data, my_dataset.exp_out, my_network.output)
            my_plot.spiral_funcion_show(my_dataset.data, my_dataset.exp_out, my_network.output)
elif dataset == 'numbers':
    my_dataset = DataGeneration()
    my_dataset.minst_numbers_traindata('Datasets/Mnist-Numbers/mnist_train.mynp')
    my_network = Model(np.array(((Dense_Layer(784, 32), Relu()),
                                (Dense_Layer(32, 64), Relu()),
                                (Dense_Layer(64, 128), Relu()),
                                (Dense_Layer(128, 128), Sigmoid()),
                                (Dense_Layer(128, 64), Relu()),
                                (Dense_Layer(64, 32), Relu()),
                                (Dense_Layer(32, 10), Soft_Max()))),
                                load_file)
    if visualize:
        tcks = [i for i in range(10)]
        my_plot = Visualize(4, 8, t_layout=1, fgsize=(12,6), xtcks=tcks, ytcks=tcks)
    if train:
        for __ in range(1000):
            my_network.train(my_dataset.data, my_dataset.exp_out, iterations=1, lr=0.000, save_file=save_file, batch_size=512)
            my_network.forward(my_dataset.data)
            my_plot.minst_numbers_show(my_dataset.puredata, my_dataset.exp_out, my_network.output, offset=30000)
elif dataset == 'newnumbers':
    my_dataset = DataGeneration()
    
    labelfiles = 'Datasets/Mnist-Numbers/t10k-labels.idx1-ubyte'
    imagefiles = 'Datasets/Mnist-Numbers/t10k-images.idx3-ubyte'
    
    my_dataset.mnist_numbers_traindata_from_ubyte(imagefiles, labelfiles)
    
    my_network = Model(np.array(((Dense_Layer(784, 32), Relu()),
                                (Dense_Layer(32, 64), Relu()),
                                (Dense_Layer(64, 128), Relu()),
                                (Dense_Layer(128, 128), Sigmoid()),
                                (Dense_Layer(128, 64), Relu()),
                                (Dense_Layer(64, 32), Relu()),
                                (Dense_Layer(32, 10), Soft_Max()))),
                                load_file)

    if visualize:
        tcks = [i for i in range(10)]
        my_plot = Visualize(4, 8, t_layout=1, fgsize=(12,6), xtcks=tcks, ytcks=tcks)
        my_network.forward(my_dataset.data)
        my_plot.minst_numbers_show(my_dataset.puredata, my_dataset.exp_out, my_network.output, offset=5000)
    
    if train:
        tcks = [i for i in range(10)]
        my_plot = Visualize(4, 8, t_layout=1, fgsize=(12,6), xtcks=tcks, ytcks=tcks)
        for __ in range(1000):
            my_network.train(my_dataset.data, my_dataset.exp_out, iterations=1, lr=0.001, save_file=save_file, batch_size=512)
            my_network.forward(my_dataset.data)
            my_plot.minst_numbers_show(my_dataset.puredata, my_dataset.exp_out, my_network.output, offset=5000)
