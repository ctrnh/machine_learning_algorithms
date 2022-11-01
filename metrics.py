import numpy as np

def accuracy(yhat, y):
    return sum(yhat == y)/len(y)