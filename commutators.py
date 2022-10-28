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
    return np.dot(La, U) - np.dot(U,La)

def test_angular_momentum_comutator(lattice_array, a:int, n=None, i=None, j=None):
    '''
    returns [L,U] - (ta)U 
    '''
    i = 0 if i == None else i
    j = 0 if j == None else j
    n = 1 if n == None else n
    comm = angular_momentum_commutator(lattice_array, a, n=n, i=i, j=j)
    ta = SU2_element.generators(a)
    if a in [1,2]:
        U = get_color_states(lattice_array, i=(i+1)%2, j=j)
    else:
        U = get_color_states(lattice_array, i=i, j=j)
    return comm - ta.matrix()[i,j]*U

def fourier_vector(lattice_array, a:int, k):
    '''
    calculates the fourier vectors, as in 'findPoints.R' in the genz writeup
    '''
    lattice_array = lattice_array[[1,1+a].append([i for i in range(5,lattice_array.shape[0])]),::]
    result = np.array([np.exp(1j * np.dot(i,k)) for i in lattice_array])
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