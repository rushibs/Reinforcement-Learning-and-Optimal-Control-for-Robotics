import math
import numpy as np

x_n = [-2, -1, 0, 1, 2]
u_n = [-1, 0, 1]
# J_n  = [None] * len(x_n)
J_n = []
N = 3


def calculate_x_vector(x_n, u_n):
    x = []
    for i in range(len(x_n)):
            for j in range(len(u_n)):
                f = x_n[i] - u_n[j] + 1
                if f <= 2 and f>= -2:
                    None
                elif f>2:
                    f = 2
                else:
                    f = -2
                x.append(f) 

    return x


def calculate_step_cost(x, u_n):
    cost = []
    for i in range(len(x)):
        for j in range(len(u_n)):
            c = 5*(abs(x[i])) + 2*(abs(u_n[j])) 
    cost.append(c)
         


while (N>=0):
    J = []
    if N == 3:  
        for k in range(len(x_n)):
            if x_n[k] == 1:
                J.append(0)
            else:
                J.append(1000)
        J_n.append(J)
        N -= 1
    else:
        x = calculate_x_vector(x_n, u_n)
           
                

