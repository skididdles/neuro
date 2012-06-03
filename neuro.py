"""
repo: https://github.com/Yannbane/neuro

Started on 03. 06. 2012., by Bane.

This program provides some basic classes for neural networks.
"""

import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
    
def step(x, threshold):
    if x > threshold:
        return 1
    else:
        return 0
    

class SLP():
    """This is the single-layer perceptron, the most basic of neural networks. It is non-linear."""
    def __init__(self, ni, no, lr):
        self.lr = lr #Learning rate
        self.ni = ni #Number of inputs
        self.no = no #Number of outputs
        
        self.inputs = [0 for i in range(self.ni)]
        self.outputs = [0 for i in range(self.no)]
        self.sums =  [0 for i in range(self.no)]
        self.thresholds =  [random.random() for i in range(self.no)]

        self.ni += 1
        self.inputs.append(1)
        self.weights = [[random.random() for i in range(self.ni)] for i in range(self.no)]
        
    def updateInputs(self, inputVector):
        for i in range(len(inputVector)):
            self.inputs[i] = inputVector[i]
    
    def calc(self, inputVector):
        if len(inputVector) != self.ni - 1:
            raise ValueError("Wrong number of inputs!")

        self.updateInputs(inputVector)
        
        for i in range(self.no):
            self.sums[i] = 0

            for j in range(self.ni):
                self.sums[i] += self.inputs[j]*self.weights[i][j]
            
            self.outputs[i] = step(self.sums[i], self.thresholds[i])

    def learn(self, targetVector):
        for i in range(self.no):
            self.thresholds[i] -= (targetVector[i] - self.outputs[i]) * self.lr
            
            for j in range(self.ni):
                self.weights[i][j] += (targetVector[i] - self.outputs[i]) * self.inputs[j] * self.lr
            
    def cycle(self, inputVector, targetVector):
        if len(targetVector) != self.no:
            raise ValueError("Wrong number of inputs!")
    
        self.calc(inputVector)
        self.learn(targetVector)
                
                
    
