import numpy as np


xn = [-2,-1,0,1,2]
un = [-1,0,1]
j = np.zeros([5,5])
u_optimum = np.zeros([5,4])
xn_array = [0, 0, 0]
xn_index = [0, 0, 0]
stage_cost = [0, 0, 0]
u_array = [0, 0, 0]

for i in range(5):
    #change terminal condition
    if xn[i] == 1:
        j[i,-1] = 0
    else:
        j[i,-1] = 1000

for i in range(4):
    for x in range(5):
        for u in range(3):
            #change system dynamics eq
            # xn1 = -xn[x]-un[u]+1
            # xn1 = -xn[x]+un[u]
            xn1 = -xn[x]+un[u]-1
            if xn1>2:
                xn1 = 2
            elif xn1<-2:
                xn1 = -2
            else:
                xn1 = xn1
            xn_index[u] = xn.index(xn1)
            xn_array[u] = xn1
            #change cost function
            # stage_cost[u] = (5*abs(xn[x])) + (5*abs(un[u]))
            # stage_cost[u] = (1*xn[x]*xn[x])+(1*abs(un[u]))
            stage_cost[u] = (2*((xn[x]) ** 2))+(abs(un[u]))
        
        jn = [(j[xn_index[0],(4-i)])+(stage_cost[0]), (j[xn_index[1],(4-i)])+(stage_cost[1]), (j[xn_index[2],(4-i)])+(stage_cost[2])]
        jn_min = np.min(jn)
        j[x, 3-i] = jn_min
        u_optimum[x,3-i] = un[jn.index(jn_min)]
        # print(un[jn.index(jn_min)])


np.set_printoptions(suppress=True)
print(j[:,1:5])
print(u_optimum[:,1:4])