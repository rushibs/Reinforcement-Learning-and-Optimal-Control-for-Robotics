#Find a sequence of states and associated cost given a control sequence

import math
import numpy as np

x = [None] * 5
u = [-1,0,-1,0] 
J = [None] * 4
x[0] = 1

for i in range(len(x)-1):
    s = -x[i] - u[i] + 1
    if s<=2 and s>=-2:
        x[i+1] = s
    elif s>2:
        x[i+1] = 2
    else:
        x[i+1] = -2 

f=0

for j in range(len(u)):
    f = f + 5*(abs(x[i])) + 5*(abs(u[i])**2)

f = f + 100*((-2-x[len(x)-1])**2)

print(x)
print(f)