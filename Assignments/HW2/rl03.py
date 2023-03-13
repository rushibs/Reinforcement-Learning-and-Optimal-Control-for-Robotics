# Write a python function next_state that computes the next state given the current state and contro

import numpy as np


def next_state(x, u):
    ## fill this function to return the next state
    x_n1 = x + 0.1*u
    if (x_n1 > 5):
        x_n1 = 5
    elif (x_n1 < 0):
        x_n1 = 0
    else:
        x_n1 = x_n1
    return x_n1