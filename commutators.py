import numpy as np
from fibonacci import generate_vertices
from derivative import angular_momentum, new_angular_momentum
from multiprocessing import Pool
from su2_element import *
# all things implementing the commutators

def angular_momentum_commutator(lattice_array, a:int, n=None, i=None, j=None, left=True, ang=angular_momentum):
    '''
    implementation of the commutator [La,U]
    '''
    i = 0 if i == None else i
    j = 0 if j == None else j
    n = 1 if n == None else n
    if n==1:
        La = ang(lattice_array, a=a, left=left)
    else:
        La = ang(lattice_array, a=a, left=left, n=n)
    U = get_color_states(lattice_array, i=i, j=j)
    return np.dot(La, U) - np.dot(U,La)

def La_Lb_commutator(lattice_array, a:int, b:int, n=None, ang=angular_momentum):
    '''
    implementation of the commutator [La, Lb] + 2i eps_{abc} Lc
    '''
    def epsilon() -> int:
        if a == b:
            return 0
        t = np.array([a,b,c])   #roll [a,b,c] s.t 1 is in front and check if its equal to [1,2,3]
        index = np.where(t == 1)[0][0]
        if np.all(np.roll(t, -index) == np.array([1,2,3])):
            return 1
        return -1

    n = 1 if n == None else n
    c = ({1,2,3} - {a,b}).pop()
    #args = [(lattice_array, 1, n),(lattice_array, 2, n),(lattice_array, 3, n)]
    #with Pool() as pool:
    #    res = pool.starmap(ang, args)
    #    pool.close()
    #La = res[0]
    #Lb = res[1]
    #Lc = res[2]
    if n == 1:
        La = ang(lattice_array, a)
        Lb = ang(lattice_array, b)
        Lc = ang(lattice_array, c)
    else:
        La = ang(lattice_array, a, n=n)
        Lb = ang(lattice_array, b, n=n)
        Lc = ang(lattice_array, c, n=n)
    return (np.dot(La, Lb) - np.dot(Lb,La)) + 2j*epsilon()*Lc


def calc_ta_U(lattice_array, a = 1, i = 0, j = 0):
    '''
    calculates (taU)_{i,j} for a given lattice array
    '''
    if a == 1:
        return get_color_states(lattice_array, i = int(not i), j = j)
    if a == 2:
        return -1j*(-1)**i * get_color_states(lattice_array, i = int(not i), j = j)
    if a == 3:
        return (-1)**i * get_color_states(lattice_array, i = i, j = j)
    raise BaseException("a is likely not 1, 2 or 3")

def test_angular_momentum_comutator(lattice_array, a:int, n=None, i=None, j=None, ang=angular_momentum):
    '''
    returns [L,U] - (ta)U 
    '''
    i = 0 if i == None else i
    j = 0 if j == None else j
    n = 1 if n == None else n
    comm = angular_momentum_commutator(lattice_array, a, n=n, i=i, j=j, ang=ang)
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
    z = np.dot(commutator, vec.T)
    z = np.absolute(z)
    return np.mean(z)



def main():
    lattice = generate_vertices(2**5)
    comm = La_Lb_commutator(lattice, 1, 2, 1, ang=new_angular_momentum)
    print(comm)
    return

if __name__ == "__main__":
    main()