import math
import numpy as np


possible_states = np.arange(0.,5.1,0.1)
possible_control = np.array([0,1,2,3,4.,5.])

def get_running_cost(a, b, x, u):
    
    running_cost =  a*(abs(x - 5)) + b*(u**2)

    return running_cost

def get_final_cost(x):
    if x == 5:
        final_cost = 0
    else:
        final_cost = 10000000

    return final_cost

def next_state(x, u):
    f = x + 0.1*u
    if f <= 5 and f >= 0:
        next_state = f
    elif f > 5:
        next_state = 5
    else:
        next_state = 0

    return next_state

def get_cost_to_go(next_cost_to_go):
    ## next_cost_to_go is a 1D numpy array of size = number of possible states
    ## fill this function (and change what is returned)
    optimal_control = np.zeros_like(next_cost_to_go)
    cost_to_go = np.zeros_like(next_cost_to_go)
   
    for x in possible_states:
        for u in possible_control:
            current_cost = get_running_cost(a,b,x,u)
            next_cost = next_cost_to_go(next_state(x,u)*10)
            running_cost = current_cost + next_cost
            if (u==0):
                mini = current_cost + next_cost
                min_control = u
            if  (running_cost < mini):
                mini = running_cost
                min_control = u

            next_cost[int(x*10)] = mini
            optimal_control[int(x*10)] = min_control
            
    return next_cost, optimal_control