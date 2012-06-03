"""
repo: https://github.com/Yannbane/neuro

Started on 03. 06. 2012., by Bane.

This program provides some basic classes for neural networks.
"""

import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
    
def step(x):
    if x > 1:
        return 1
    else:
        return 0
    

class SLP():
    """This is the single-layer perceptron, the most basic of neural networks. It is non-linear."""
    def __init__(self, ni, no, lr, af):
        self.lr = lr #Learning rate
        self.ni = ni #Number of inputs
        self.no = no #Number of outputs
        self.af = af #The activation function
        
        self.inputs = [0 for i in range(self.ni)]
        self.outputs = [0 for i in range(self.no)]
        self.sums =  [0 for i in range(self.no)]
        self.weights = [[random.random() for i in range(self.ni)] for i in range(self.no)]
        
    def calc(self, inputVector):
        
        if len(inputVector) != self.ni:
            raise ValueError("Wrong number of inputs!")

        self.inputs = inputVector
        
        for i in range(self.no):
            self.sums[i] = 0

            for j in range(self.ni):
                self.sums[i] += self.inputs[j]*self.weights[i][j]
            
            self.outputs[i] = self.af(self.sums[i])

    def learn(self, targetVector):
        for i in range(self.no):
            for j in range(self.ni):
                self.weights[i][j] += (targetVector[i] - self.outputs[i]) * self.inputs[j] * self.lr
            
    def cycle(self, inputVector, targetVector):
        if len(targetVector) != self.no:
            raise ValueError("Wrong number of inputs!")
    
        self.calc(inputVector)
        self.learn(targetVector)
                
                
