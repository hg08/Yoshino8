import numpy as np

# Functions
def init_S(M,L,N):
    S = np.ones((M,L+1,N))
    for i in range(M):
        for j in range(L+1):
            S[i,j,:] = generate_coord(N)
    print("[For testing the validation of S] Ratio of positive S and negative S: {:7.6f}".format(S[S<0].size / S[S>0].size))
    return S
def generate_coord(N):
    """Randomly set the initial coordinates,whose components are 1 or -1."""
    v = np.ones(N)
    list_spin = [-1,1]
    for i in range(N):
        v[i] = np.random.choice(list_spin)
    return v
