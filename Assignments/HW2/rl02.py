from typing import final
import numpy as np

# a is defined a priori, you may use a = 2. to test your code
# b is defined a priori, you may use b = 10. to test your code

def get_running_cost(a, b, x, u):
    ## fill this function to return the running cost
    running_cost = (a*abs(x -5)) + (b*u*u)
    return running_cost

def get_final_cost(x):
    ## fill this function to return the final cost given x
    if(x == 5):
        final_cost = 0
    else:
        final_cost = 10000000
    return final_cost