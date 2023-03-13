import numpy as np
import matplotlib.pyplot as plt 
import math

# hint: the two functions should work for states and control vectors of arbitrary dimensions (i.e. do not hard code 4 and 2!)
# you may want to test individual functions and the whole code first on your computer for easier debugging and to find appropriate gain matrices


def solve_LQR_trajectory(A, B, Q, R, x_bar, N):
    '''
    A, B, Q and R are the matrices defining the OC problem
    x_bar is the trajectory of desired states of size dim(x) x (N+1)
    N is the horizon length
    
    The function returns 1) a list of gains of length N and 2) a list of feedforward controls of length N
    '''
    K_gains = []
    k_feedforward = []
    list_of_P = [] # K will be from 1 to N-1
    list_of_p = [] # P will be from 1 to N

    list_of_P.append(Q)
    list_of_p.append(np.matmul(-Q, x_bar[:,-1]))

    for i in range(N):
                
        ##Linear FeedBack
        K_temp1 = np.linalg.inv((np.matmul(B.T, np.matmul(list_of_P[i], B))) + R)
        K_temp2 = np.matmul(B.T, np.matmul(list_of_P[i], A))

        K_n = np.matmul(-K_temp1, K_temp2)

        P_temp1 = Q
        P_temp2 = np.matmul(A.T, np.matmul(list_of_P[i], A))
        P_temp3 = np.matmul(A.T, np.matmul(list_of_P[i], np.matmul(B, K_n)))
        P_n = P_temp1 + P_temp2 + P_temp3

        K_gains.append(K_n)
        list_of_P.append(P_n)

        ##FeedForward
        k_temp1 = np.linalg.inv(np.matmul(np.transpose(B), np.matmul(list_of_P[i], B)) + R)
        k_temp2 = np.matmul(np.transpose(B), list_of_p[i])
        k_n = np.matmul(-k_temp1, k_temp2)

        qn = np.matmul(-Q, x_bar[:,N-i-1])
        
        p_temp1 = qn
        p_temp2 = np.matmul(np.transpose(A), list_of_p[i])
        p_temp3 = np.matmul(np.transpose(A), np.matmul(list_of_P[i], np.matmul(B, k_n)))
        p_n = p_temp1 + p_temp2 + p_temp3

        k_feedforward.append(k_n)
        list_of_p.append(p_n)


    K_gains.reverse()
    k_feedforward.reverse()

    return K_gains, k_feedforward
    
def simulate_dynamics(A, B, K_gains, k_feedforward, x0, N):
    '''
    A, B define the system dynamics
    K_gains is a list of feedback gains of length N
    k_feedforward is a list of feedforward controls of length N
    x0 is the initial state (array of dim (dim(state) x 1))
    
    The function returns 1) an array of states (dim(states) x N+1) and 2) an array of controls (dim(control) x N)
    '''
    x = np.zeros([A.shape[0], N+1])
    u = np.zeros([B.shape[1], N])
    x[:,0] = x0[:,0]

    for i in range(N):
        u_n = np.matmul(K_gains[i], x[:,i]) + k_feedforward[i]
        # print(un.shape)
        u[:,i] = u_n

        x_n = x[:,i]
        x_n1 = np.matmul(A, x_n) + np.matmul(B,u_n)
        # print(xn1.shape)
        x[:,i+1]=x_n1

    return x, u

# we generate a random initial state
x0 = np.random.uniform(-2.,.2,(4,1))

deltaT = 0.01

# we want a trajectory of 20 seconds
t = np.arange(0.,20.01, deltaT)
N = len(t)-1

omega = 0.5*np.pi

### WRITE CODE THAT SOLVES THE PROBLEM HERE ###
A = np.diag([1.0, 1.0, 1.0, 1.0])
A[0,1] = deltaT
A[2,-1] = deltaT

B = np.zeros([4,2])
B[1,0] = deltaT
B[-1,-1] = deltaT

B = np.zeros([4,2])
B[1,0] = deltaT
B[-1,-1] = deltaT

x_bar = np.zeros((A.shape[0], N+1))
# x_bar[:,0] = x0[:,0]

for i in range(N+1):
    T = 0.01*(i)
    x_bar_1 = np.array([np.sin(0.5*np.pi*T), 
                        0.5*np.pi*np.cos(0.5*np.pi*T),
                        np.sin(2*0.5*np.pi*T), 
                        2*0.5*np.pi*np.cos(2*0.5*np.pi*T)])
    x_bar[:,i] = x_bar_1

x_bar = x_bar.reshape(4,2002)

Q = 100*np.diag([1.0, 1.0, 1.0, 1.0])
R = 2*np.diag([0.1, 0.1])


K_gains, k_feedforward = solve_LQR_trajectory(A, B, Q, R, x_bar, N)
x, u = simulate_dynamics(A, B, K_gains, k_feedforward, x0, N)


#### ONCE THIS IS DONE WE PLOT THE RESULTS ####
plt.figure()
plt.plot(x[0,:], x[2,:], x_bar[0,:], x_bar[2,:])
plt.xlabel('Position in first dimension')
plt.ylabel('Position in second dimension')
plt.title('Trajectory of the car in 2D')
plt.legend(['simulated trajectory','desired'])

plt.figure()
names = ['Pos. 1', 'Vel. 1', 'Pos. 2', 'Vel. 2']
for i in range(4):
    plt.subplot(4,1,1+i)
    plt.plot(t,x[i,:], t, x_bar[i,:], '--')
    plt.ylabel(names[i])
plt.xlabel('Time [s]')
    
plt.figure()
plt.plot(t[:-1], u.T)
plt.legend(['Control 1', 'Control 2'])
plt.xlabel('Time [s]')

plt.show()