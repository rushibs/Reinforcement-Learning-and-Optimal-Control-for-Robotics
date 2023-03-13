import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(suppress=True)


# N is defined a priori, you may use N = 25 to test your code
# a is defined a priori, you may use a = 2. to test your code
# b is defined a priori, you may use b = 10. to test your code

N = 25
a = 2
b = 10

## We print the unknown variables just to know
print(f'The problem instance uses a={a}, b={b} and a horizon length of N={N}')

# this numpy array enumerates all possible states
possible_states = np.arange(0.,5.1,0.1)

# this numpy array enumerates all possible control
possible_control = np.array([0,1,2,3,4.,5.])


## HERE WRITE THE CODE TO SOLVE THE PROBLEM ##

cost_to_go = np.zeros((possible_states.size, N+1))
optimal_control = np.zeros((possible_states.size, N))
u = np.zeros(N)
x = np.zeros(N+1)
J = 0

def get_next_state(x_n, u_n):
    x_n1 = x_n + (0.1*u_n)
    if (x_n1 > 5):
        x_n1 = 5.0
    elif (x_n1 < 0):
        x_n1 = 0.0
    else:
        x_n1 = round(x_n1, 2)
    return x_n1

def get_running_cost(a, b, x_n, u_n):
    running_cost = (a*abs(x_n -5)) + (b*u_n*u_n)
    return running_cost

for x2 in range(possible_states.size):
    x_n = possible_states[x2]
    if x_n == 5.0:
        cost_to_go[x2,-1] = 0
    else:
        cost_to_go[x2,-1] = 10000000


for n in range(N):
    for x1 in range(possible_states.size):
        x_n = possible_states[x1]
        stage_cost = np.zeros(possible_control.size)
        for u1 in range(possible_control.size):
            u_n = possible_control[u1]
            x_n1 = get_next_state(x_n, u_n)

            possible_states1 = np.around(possible_states, 2)
            possible_states_list = possible_states1.tolist() 

            index_ = possible_states_list.index(x_n1)  
            next_cost = cost_to_go[index_, N-n]
            stage_cost[u1] = get_running_cost(a, b, x_n, u_n) + next_cost

        min_cost_index = np.argmin(stage_cost)
        cost_to_go[x1, N-1-n] = stage_cost[min_cost_index]
        optimal_control[x1, N-1-n] = possible_control[min_cost_index]


for i in range(N):
    possible_states1 = np.around(possible_states, 2)
    possible_states_list = possible_states1.tolist() 
    x_index = possible_states_list.index(x[i])

    x_n = possible_states1[x_index]
    u[i] = optimal_control[x_index,i]
   
    x[i+1] = get_next_state(x_n, u[i])
    print(x[i+1], x_n, u[i])

J = cost_to_go[0,0]

## Once the code finds the right answer, we plot the resulting x and u
plt.figure()
plt.subplot(2,1,1)
plt.plot(x, '-o')
plt.ylabel('State')
plt.subplot(2,1,2)
plt.plot(u, '-o')
plt.ylabel('Control')
plt.xlabel('Stages')
# plt.show()

## we also print the optimal cost
print(f'The optimal cost found is J={J}')