import numpy as np

# hint: to test your code you can test your function on the A,B,Q,R matrices shown in class
A=np.array([[ 0.,   0.,   0.,   0. ],
            [ 0.,  -1.,   0.,  -1.5],
            [ 0.,  -1.,   0.,   0. ],
            [-2.,   0.,  -1.,  -1. ]])

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
        k_temp1 = np.linalg.inv((np.matmul(B.T, np.matmul(list_of_P[i], B))) + R)
        k_temp2 = np.matmul(B.T, np.matmul(list_of_P[i], A))

        Kn = np.matmul(-k_temp1, k_temp2)

        p_temp1 = Q
        p_temp2 = np.matmul(np.transpose(A), np.matmul(list_of_P[i], A))
        p_temp3 = np.matmul(np.transpose(A), np.matmul(list_of_P[i], np.matmul(B, Kn)))
        Pn = p_temp1 + p_temp2 + p_temp3

        list_of_K.append(Kn)
        list_of_P.append(Pn)

    list_of_K.reverse()
    list_of_P.reverse()

    return list_of_P, list_of_K

P, K = solve_LQR(A, B, Q, R, QN, N)
print(P[0])
print(K[0])

print(P[-1])
print(K[-1])