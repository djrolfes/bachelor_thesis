import numpy as np
from numba import njit
from itertools import combinations
from su2_element import SU2_element, su2_product
from lattice_actions import *
from multiprocessing import Pool
from scipy import sparse

#@njit
def is_linear_independent(matrix, eps=1e-6):
    '''
    returns if the vectors making up a given matrix are linear independent, by calculating the eigenvalues
    of the matrix and checking wether one is zero. 
    '''
    return abs(np.linalg.det(matrix)) > eps
        
#@njit
def get_conn_elements(element : SU2_element, neighbors, left=True):
    '''
    calculate the SU2_elements connecting the neighbor elements with the given element
    '''
    if left:
        return np.array([su2_product(elem, element.inverse()) for elem in neighbors])
    else:
        return np.array([su2_product(element.inverse(), elem) for elem in neighbors])


def calc_alpha(element: SU2_element, neighbor: SU2_element, left=True):
    '''
    returns the alpha-vector for a given element U and neighbor V
    left = True:    V_i \dot U^-1
    left = False:   U^-1 \dot V_i
    '''
    if left:
        conn = su2_product(neighbor, element.inverse())
    else:
        conn = su2_product(element.inverse(), neighbor)
    return conn.get_angles()


def get_linear_independent(element : SU2_element, neighbors, left=True):
    '''
    returns the (alpha, alpha, alpha) matrix to calculate gamma_i for the first three viable neighbors
    neighbors is an ordered array of SU2_element objects
    '''
    i=3
    alpha_matrix = np.zeros((3,3))
    while i<neighbors.shape[0]:
        combs = combinations(neighbors[1:i+1], r=3)
        index_combs = np.array(list(combinations(range(1,i+1), r=3)))
        for indeces, comb in zip(index_combs, combs):
            for j in range(3):
                alpha_matrix[:,j] = calc_alpha(element, comb[j], left=left).T
            if is_linear_independent(alpha_matrix):
                return alpha_matrix, indeces 
        i += 1 
    else:
        raise BaseException("No linear independent combination passible")
    
#@njit
def calc_gamma(alpha_matrix, a: int):
    '''
    solves e_a = gamma \dot alpha_matrix for gamma using np.linalg.solve
    '''
    unit_vec = np.zeros(3)
    unit_vec[a-1] = 1
    return np.linalg.solve(alpha_matrix, unit_vec.T)


def angular_momentum_two_neighbors_new(lattice_array, a: int, n=None, left=True):
    '''
    implementation of the forward derivative with 2x3 neighbors using a different approximation
    '''
    n = 2
    La = np.zeros((lattice_array.shape[0], lattice_array.shape[0]))
    su2_lattice = SU2_element.vectorize_init(lattice_array)

    for index, element in enumerate(lattice_array):
        _, neighbors_indeces = get_neighbors_single_element(element, lattice_array)
        su2_neighbors = su2_lattice[neighbors_indeces]
        alpha_matrix, inds = get_linear_independent(SU2_element(element), su2_neighbors, left=left)
        gammas = calc_gamma(alpha_matrix, a)
        La[index, index] += -np.sum(gammas)
        for i,ind in enumerate(inds):
            La[index, neighbors_indeces[ind]] += gammas[i]
        su2_neighbors = np.delete(su2_neighbors, inds, axis=0)
        alpha_matrix, inds = get_linear_independent(SU2_element(element), su2_neighbors, left=left)
        gammas = calc_gamma(alpha_matrix, a)
        La[index, index] += np.sum(gammas)
        for i,ind in enumerate(inds):
            La[index, neighbors_indeces[ind]] += gammas[i]

    return -1j*La/n


def angular_momentum(lattice_array, a: int, n=None, left=True):
    '''
    implementation of the forward derivative with nx3 neighbors
    '''
    n = 1 if n == None else n
    La = np.zeros((lattice_array.shape[0], lattice_array.shape[0]))
    su2_lattice = SU2_element.vectorize_init(lattice_array)

    for index, element in enumerate(lattice_array):
        _, neighbors_indeces = get_neighbors_single_element(element, lattice_array)
        su2_neighbors = su2_lattice[neighbors_indeces]
        for _ in range(n):
            alpha_matrix, inds = get_linear_independent(SU2_element(element), su2_neighbors, left=left)
            gammas = calc_gamma(alpha_matrix, a)
            La[index, index] += -np.sum(gammas)
            for i,ind in enumerate(inds):
                La[index, neighbors_indeces[ind]] += gammas[i]
            su2_neighbors = np.delete(su2_neighbors, inds, axis=0)
    return -1j*La/n

def new_get_linear_independent(element : SU2_element, neighbors, step=None, left=True):
    '''
    gets a linear independent matrix of alphas for new_angular_momentum
    '''
    step = 1 if step==None else step
    if step == 1:
        return get_linear_independent(element, neighbors, left=left)
    inds = list(range((3*(step+1))))[1::step]
    inds = inds[:3]

    alpha_matrix = np.zeros((3,3))
    i=3
    while i<neighbors.shape[0]:
        for j in range(3):
            alpha_matrix[:,j] = calc_alpha(element, (neighbors[inds])[j], left=left).T
        if is_linear_independent(alpha_matrix):
            return alpha_matrix, inds 
        i += 1
        inds[-(i%3)] += 1
    raise BaseException("No linear independent combination passible")


def wrap(args):
    index, element, su2_lattice, lattice_array,a,n,left = args 
    La_i = np.zeros(lattice_array.shape[0])
    _, neighbors_indeces = get_neighbors_single_element(element, lattice_array)
    #su2_lattice = SU2_element.vectorize_init(lattice_array)
    su2_neighbors = su2_lattice[neighbors_indeces]        
    for neighbor_group in range(n):     
        alpha_matrix, inds = new_get_linear_independent(SU2_element(element), su2_neighbors, step=(n-neighbor_group), left=left)
    gammas = calc_gamma(alpha_matrix, a)
    La_i[index] = -np.sum(gammas)
    for i,ind in enumerate(inds):
        La_i[neighbors_indeces[ind]] = gammas[i]
    su2_neighbors = np.delete(su2_neighbors, inds, axis=0)
    return index,La_i

def mp_new_angular_momentum(lattice_array, a: int, n=None, left=True):
    '''
    implementation of the derivative with nx3 neighbors, with a different way of choosing neighbors using multiprocessing
    '''
    n = 1 if n == None else n
    La = np.zeros((lattice_array.shape[0], lattice_array.shape[0]))
    su2_lattice = SU2_element.vectorize_init(lattice_array)

    args = list()
    for index, element in enumerate(lattice_array):
        args.append(((index, element, su2_lattice, lattice_array,a,n,left)))

    with Pool(4) as pool:
        chunksize = 500 if int(lattice_array.shape[0]/4+1)<500 else int(lattice_array.shape[0]/4+1)
        print(chunksize)
        result = pool.imap_unordered(wrap, args, chunksize=chunksize)
        pool.close()

        for _, res in enumerate(result):
            La[res[0],:] = res[1]
    return -1j* La/n


def new_angular_momentum(lattice_array, a: int, n=None, left=True):
    '''
    implementation of the derivative with nx3 neighbors, with a different way of choosing neighbors
    '''
    n = 1 if n == None else n
    La = np.zeros((lattice_array.shape[0], lattice_array.shape[0]))
    su2_lattice = SU2_element.vectorize_init(lattice_array)
    
    for index, element in enumerate(lattice_array):
        _, neighbors_indeces = get_neighbors_single_element(element, lattice_array)
        su2_neighbors = su2_lattice[neighbors_indeces]        
        for neighbor_group in range(n):     
            alpha_matrix, inds = new_get_linear_independent(SU2_element(element), su2_neighbors, step=(n-neighbor_group), left=left)
            gammas = calc_gamma(alpha_matrix, a)
            La[index, index] += -np.sum(gammas)
            for i,ind in enumerate(inds):
                La[index, neighbors_indeces[ind]] += gammas[i]
            su2_neighbors = np.delete(su2_neighbors, inds, axis=0)
    return -1j* La/n

def new_angular_momentum_sparce(lattice_array, a: int, n=None, left=True):
    '''
    implementation of the derivative with nx3 neighbors, with a different way of choosing neighbors
    '''
    n = 1 if n == None else n
    La = np.zeros((lattice_array.shape[0], lattice_array.shape[0]))
    La = sparse.lil_matrix(La)
    su2_lattice = SU2_element.vectorize_init(lattice_array)
    
    for index, element in enumerate(lattice_array):
        _, neighbors_indeces = get_neighbors_single_element(element, lattice_array)
        su2_neighbors = su2_lattice[neighbors_indeces]        
        for neighbor_group in range(n):     
            alpha_matrix, inds = new_get_linear_independent(SU2_element(element), su2_neighbors, step=(n-neighbor_group), left=left)
            gammas = calc_gamma(alpha_matrix, a)
            La[index, index] += -np.sum(gammas)
            for i,ind in enumerate(inds):
                La[index, neighbors_indeces[ind]] += gammas[i]
            su2_neighbors = np.delete(su2_neighbors, inds, axis=0)
    return -1j* La/n


def main():
    lattice = generate_vertices(2**8)
    t1 = mp_new_angular_momentum(lattice,1)
    t2 = new_angular_momentum(lattice, 1)
    print(t1==t2)
    
    
    return

if __name__ == "__main__":
    main()