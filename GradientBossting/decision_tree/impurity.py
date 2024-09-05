import numpy as np


def impurity(left_y, right_y, criterion):

    y = np.hstack((left_y, right_y))

    return criterion(y) - criterion(left_y)*(left_y.size/ y.size) - \
           criterion(right_y)*(right_y.size/ y.size) 