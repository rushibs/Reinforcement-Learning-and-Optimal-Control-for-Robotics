import numpy as np

# a is defined a priori, you may use a = 2. to test your code
# b is defined a priori, you may use b = 10. to test your code

# this numpy array enumerates all possible states (assume that your cost to go will return a numpy array that follows this ordering)
possible_states = np.arange(0.,5.1,0.1)

# this numpy array enumerates all possible control
possible_control = np.array([0,1,2,3,4.,5.])

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

def next_state(x, u):
    ## fill this function to return the next state
    x_n1 = x + 0.1*u
    if (x_n1 > 5):
        x_n1 = 5.0
    elif (x_n1 < 0):
        x_n1 = 0.0
    else:
        x_n1 = round(x_n1, 2)
    return x_n1

def get_cost_to_go(next_cost_to_go):
    ## next_cost_to_go is a 1D numpy array of size = number of possible states
    ## fill this function (and change what is returned)
    optimal_control = np.zeros_like(next_cost_to_go)
    cost_to_go = np.zeros_like(next_cost_to_go)
    
    for x in range(possible_states.size):
        x_n = possible_states[x]
        stage_cost = np.zeros(possible_control.size)
        for u in range(possible_control.size):
            u_n = possible_control[u]
            x_n1 = next_state(x_n, u_n)
            a = 2
            b = 10          
            stage_cost[u] = get_running_cost(a, b, x_n, u_n) + next_cost_to_go[int(x_n1*10)]
        
        min_cost = np.min(stage_cost)
        cost_to_go[x] = min_cost
        optimal_control[x] = possible_control[np.where(stage_cost == min_cost)]

    return cost_to_go, optimal_control

next_cost_to_go = np.asarray([23.95094599, 98.47135396, 57.59335839, 87.82554739, 34.3312179, 56.13375334,
 26.57064696, 44.44871598, 82.71420876, 74.52716422,  3.53115065, 77.63775305,
 12.40739601, 36.4595957,  59.40233534,  4.70819952, 21.46591003, 57.0879265,
 81.61846371,  0.48279419, 10.53260326, 73.39517583, 98.32879289, 70.40842735,
 35.82546607, 36.80780805, 19.18859102, 95.06372384, 39.61033645, 40.04696987,
 44.05455346, 64.51422395, 20.42412485, 89.96286008,  7.52534912, 38.53990917,
 61.59362331, 63.80259812, 85.8456811,  61.30329262, 38.66895697, 79.58696853,
 62.91652601, 90.48897514,  0.66300979, 23.72563087, 26.69136015, 34.30190962,
 77.60946838, 22.46591977, 83.14574476])

cost_to_go, optimal_control = get_cost_to_go(next_cost_to_go)
print(cost_to_go)
print(optimal_control)