import numpy as np
import matplotlib.pyplot as plt

# hint: to test your code you can test your function on the A,B,Q,R matrices shown in class
A=np.array([[ 0.,   0.,   0.,   0. ],
            [ 0.,  -1.,   0.,  -1.5],
            [ 0.,  -1.,   0.,   0. ],
            [-2.,   0.,  -1.,  -1. ]])

x0 = np.transpose(np.array([1.2, 1, 0.5,0.6]))

B=np.array([[ 0.5,  0. ],
            [ 0.,  -0.5],
            [ 0.,   0. ],
            [ 0.,   0. ]])

Q=np.array([[ 297.03961416,  -86.34661712,  -22.90563286,   92.7355567 ],
            [ -86.34661712,  533.24451064,  -82.79574115, -129.15744318],
            [ -22.90563286,  -82.79574115,  445.89479039,   71.47409607],
            [  92.7355567,  -129.15744318,   71.47409607,  563.91469176]])

R=np.array([[ 31.02531112,   5.15342942],
            [  5.15342942, 324.71944137]])

QN=np.array([[ 833.71280299, -173.54626585,  176.8133831,    42.04782892],
            [-173.54626585,  452.16374423,  283.97797782,  -71.08087348],
            [ 176.8133831,   283.97797782,  491.14472847,  -17.2919489 ],
            [  42.04782892,  -71.08087348,  -17.2919489,   684.95596253]])

N = 13


def solve_LQR(A, B, Q, R, QN, N):
    '''
    A, B, Q and R are the matrices defining the OC problem
    QN is the matrix used for the terminal cost
    N is the horizon length
    '''
    list_of_P = [] # K will be from 1 to N-1
    list_of_K = [] # P will be from 1 to N

    list_of_P.append(QN)

    for i in range(N):
        K_1 = (np.matmul(np.transpose(B), np.matmul(list_of_P[i], B))) + R
        K_1_inv = np.linalg.inv(K_1)
        K_2 = np.matmul(np.transpose(B), np.matmul(list_of_P[i], A))

        Kn = np.matmul(-K_1_inv, K_2)

        P_1 = Q
        P_2 = np.matmul(np.transpose(A), np.matmul(list_of_P[i], A))
        P_3 = np.matmul(np.transpose(A), np.matmul(list_of_P[i], np.matmul(B, Kn)))
        Pn = P_1 + P_2 + P_3

        list_of_K.append(Kn)
        list_of_P.append(Pn)

    list_of_K.reverse()
    list_of_P.reverse()

    return list_of_P, list_of_K


def optimal_values(x0, P, K, A, B, N):
    x_optimal = []
    u_optimal = []
    Jn = []

    x_optimal.append(x0)

    for i in range(N):
        Xn = np.array(x_optimal[i])
        Un = np.matmul(K[i], Xn)

        Xn1 = np.matmul(A, Xn) + np.matmul(B, Un)

        J = np.matmul(np.transpose(Xn), np.matmul(P[i], Xn))

        x_optimal.append(Xn1)
        u_optimal.append(Un)
        Jn.append(J)        
        
    x_optimal = np.transpose(np.array(x_optimal))
    u_optimal = np.transpose(np.array(u_optimal))
    Jn = np.array(Jn)
    

    return x_optimal, u_optimal, Jn

P, K = solve_LQR(A, B, Q, R, QN, N)
# u_optimal, x_optimal, Jn =  optimal_values(P, K, x0, N, A, B)

x_optimal, u_optimal, Jn = optimal_values(x0, P, K, A, B, N)

x_optimal = x_optimal[0]
u_optimal = u_optimal[0]

print(np.array(P).shape)
print(np.array(K).shape)

J_optimal = Jn[0]

print(x_optimal.T.shape)

## Once the code finds the right answer, we plot the resulting x and u
plt.figure()
plt.subplot(2,1,1)
plt.plot(x_optimal.T, '-o')
plt.ylabel('States')
plt.subplot(2,1,2)
plt.plot(u_optimal.T, '-o')
plt.ylabel('Controls')
plt.xlabel('Stages')

plt.show()

# ## we also print the optimal cost
# print(f'The optimal cost found is J_optimal={J_optimal}')


# # print(P[0])
# # print(K[0])

# # print(P[-1])
# # print(K[-1])

# print(u_optimal.shape)
# print(x_optimal.shape)
print(J_optimal)