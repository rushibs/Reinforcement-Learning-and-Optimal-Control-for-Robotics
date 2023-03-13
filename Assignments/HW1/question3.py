#Find the optimal cost-to-go given optimal sequence of actions

import math
import numpy as np

x = [None] * 5 #put the no. of states
u = [0,0,1,-1] #put the controls in reverse order
J = [None] * 5
J[0] = 0 #J is also in reverse order so put value of J_n

x[4] = 2
x[0] = 0   #both are given but put in reverse order

for i in range(len(x)-1):
    x[i+1] = -x[i] - u[i]    #state function solving for x_n using x_n+1
    J[i+1] = abs(x[i+1]) + 10*(abs(u[i])**2) + J[i]        #cost function solving for J_n using J_n+1

print(J)
print(J[4])