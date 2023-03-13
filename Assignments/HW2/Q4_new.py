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

cost_to_go = np.zeros((len(possible_states), N+1))
optimal_control = np.zeros((len(possible_states), N))
u = np.zeros(N)
x = np.zeros(N+1)
J = 0

def next_state(x, u):
    f = x + 0.1*u
    if f < 0:
        next_state = f
    elif f > 5:
        next_state = 5
    else:
        next_state = round(f, 2)

    return next_state

def get_running_cost(a, b, x, u):
    
    running_cost =  a*(abs(x - 5)) + b*(u**2)

    return running_cost

for i in range(len(possible_states)):
    x_n = possible_states[i]
    if x_n == 5.0:
        cost_to_go[i,-1] = 0
    else:
        cost_to_go[i,-1] = 10000000


for n in range(N):
    for j in range(len(possible_states)):
        x_n = possible_states[j]
        stage_cost = np.zeros(len(possible_control))
        for k in range(len(possible_control)):
            u_n = possible_control[k]
            x_n1 = next_state(x_n, u_n)

            other_states = np.around(possible_states, 2)
            other_states_list = other_states.tolist() 

            pointer = other_states_list.index(x_n1)  
            next_cost = cost_to_go[pointer, N-n]
            stage_cost[k] = get_running_cost(a, b, x_n, u_n) + next_cost

        min_cost_ptr = np.argmin(stage_cost)
        cost_to_go[j, N-1-n] = stage_cost[min_cost_ptr]
        optimal_control[j, N-1-n] = possible_control[min_cost_ptr]


for i in range(N):
    other_states = np.around(possible_states, 2)
    other_states_list = other_states.tolist() 
    x_index = other_states_list.index(x[i])

    x_n = other_states[x_index]
    u[i] = optimal_control[x_index,i]
   
    x[i+1] = next_state(x_n, u[i])
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