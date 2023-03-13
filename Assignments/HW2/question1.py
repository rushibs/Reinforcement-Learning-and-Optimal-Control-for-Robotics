#Write a function for running cost and one for final cost for given x, u, a and b

import math
import numpy as np

a= 2
b = 10

def get_running_cost(a, b, x, u):
    
    running_cost =  a*(abs(x - 5)) + b*(u**2)

    return running_cost

def get_final_cost(x):
    if x == 5:
        final_cost = 0
    else:
        final_cost = 10000000

    return final_cost

