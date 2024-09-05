import numpy as np

def gini(y):
    
    _, counts = np.unique(y, return_counts=True)

    p = counts / y.size

    return (p*(1 - p)).sum()

def entropy(y):
    
    _, counts = np.unique(y, return_counts=True)

    p = counts / y.size

    return (-p*np.log2(p)).sum()