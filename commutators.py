import numpy as np
from fibonacci import generate_vertices
from derivative import angular_momentum
from su2_element import *
# all things implementing the commutators

def angular_momentum_commutator(lattice_array, a:int, n=None, i=None, j=None):
    '''
    implementation of the comutator [La,U]
    '''
    i = 0 if i == None else i
    j = 0 if j == None else j
    n = 1 if n == None else n
    La = angular_momentum(lattice_array, a, n=n)
    U = get_color_states(lattice_array, i=i, j=j)
    #print(La, "\n", U, "\n", lattice_array)
    return np.dot(La, U) - np.dot(U,La)

def calc_ta_U(lattice_array, a = 1, i = 0, j = 0):
    '''
    calculates (taU)_{i,j} for a given lattice array
    '''
    if a == 1:
        return get_color_states(lattice_array, i = int(not i), j = j)/2
    if a == 2:
        return -1j*(-1)**i*get_color_states(lattice_array, i = int(not i), j = j)/2
    if a == 3:
        return (-1)**i * get_color_states(lattice_array, i = i, j = j)/2
    raise BaseException("a is likely not 1, 2 or 3")

def test_angular_momentum_comutator(lattice_array, a:int, n=None, i=None, j=None):
    '''
    returns [L,U] - (ta)U 
    '''
    i = 0 if i == None else i
    j = 0 if j == None else j
    n = 1 if n == None else n
    comm = angular_momentum_commutator(lattice_array, a, n=n, i=i, j=j)
    ta_U = calc_ta_U(lattice_array, a = a, i = i, j = j)
    return comm - ta_U

def fourier_vector(lattice_array, a:int, k):
    '''
    calculates the fourier vectors, as in 'findPoints.R' in the genz writeup
    '''
    lattice_array = SU2_element.vectorize_init(lattice_array)
    lattice_array = lattice_array[[0,a].append([i for i in range(4,lattice_array.shape[0])])][0]
    result = np.array([np.exp(1j * np.dot(i.get_angles(),k)) for i in lattice_array])
    return result


def calc_r(commutator, vec=None):
    '''
    calculates r = 1/N * sum(abs(z_i)) as written in the writeup
    '''
    #vec = np.ones(commutator.shape[0], dtype=np.complex64) if vec == None else vec
    z = np.dot(commutator, vec.T)
    z = np.absolute(z)
    return np.mean(z)



def main():
    lattice = generate_vertices(2**4)
    t = test_angular_momentum_comutator(lattice, 1, n=1, j=0,i=0)
    print(t)
    return

if __name__ == "__main__":
    main()