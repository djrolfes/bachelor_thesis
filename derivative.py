from tkinter import N
import numpy as np
#from numba import njit
from itertools import combinations
from su2_element import SU2_element, su2_product
from fibonacci import generate_vertices
from lattice_actions import *

def swap_elements(arr, i1, i2):
    '''
    swaps two elements of an array at indeces i1 and i2
    '''
    tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp
    return arr

#@njit
def is_linear_independent(matrix, eps=1e-6):
    '''
    returns if the vectors making up a given matrix are linear independent, by calculating the eigenvalues
    of the matrix and checking wether one is zero. 
    '''
    #eps = np.finfo(np.linalg.norm(matrix).dtype).eps
    #TOLERANCE = max(eps * np.array(matrix.shape))
    #print(TOLERANCE)
    return abs(np.linalg.det(matrix)) > eps
        

def get_linear_independent_single_element(element: SU2_element, neighbors: SU2_element, left=True):
    '''
    the get_linear_independent function for a single element
    '''
    i = 4
    while i<len(neighbors):
        combs = combinations(neighbors[:i], r=3)
        index_combs = np.array(list(combinations(range(0,i), r=3)))#
        for index,comb in enumerate(combs):
            #connecting_elements = np.array([su2_product(elem, element.inverse()) for elem in comb])
            connecting_elements = get_conn_elements(element, comb, left=left)
            alphas = np.array([connecting_element.get_angles() for connecting_element in connecting_elements])
            matrix = np.column_stack(alphas)
            if is_linear_independent(matrix) == True:
                return matrix,index_combs[index]
        i+=1
    else:
        raise BaseException("No linear independent combination passible")

def get_conn_elements(element : SU2_element, neighbors, left=True):
    '''
    calculate the SU2_elements connecting the neighbor elements with the given element
    '''
    if left:
        return np.array([su2_product(elem, element.inverse()) for elem in neighbors])
    else:
        return np.array([su2_product(element.inverse(), elem) for elem in neighbors])

def angular_momentum_single_element(element, lattice_array, a: int, n=None, neighbors=None, left=True):
    '''
    calculates the angular momentum operator La by using the 3n nearest neighbors 
    of a given lattice element, should be correct

    use left=False to generate the Ra operator
    '''
    n = 1 if n == None else n
    unit_vec = np.array([0,0,0])
    unit_vec[a-1] = 1
    unit_vec = np.asmatrix(unit_vec)
    if neighbors == None:
        neighbors, sort_indeces = get_neighbors_single_element(element, lattice_array)
    #tbd: calc sort_indeces from given neighbors and the latticearray

    element = SU2_element(element)
    element_index = sort_indeces[0]
    SU2_neighbors = SU2_element.vectorize_init(lattice_array)[sort_indeces][1::]

    La = np.zeros(len(SU2_neighbors)+1)
    gammas = np.array([])
    for neighbor_group in range(n):
        connecting_elements = get_conn_elements(element, SU2_neighbors[:3], left=left)
        alphas = [connecting_element.get_angles() for connecting_element in connecting_elements]
        matrix = np.asmatrix(np.column_stack(alphas))
        if not is_linear_independent(matrix): #somewhat tested
            #print('linear dep', matrix, "\n", np.linalg.svd(matrix),"\n", np.linalg.eig(matrix), "\n\n")
            matrix, swap_indeces = get_linear_independent_single_element(element, SU2_neighbors, left=left)

            for index, num in enumerate(swap_indeces):
                sort_indeces = swap_elements(sort_indeces, index+neighbor_group, num+neighbor_group)
                SU2_neighbors = swap_elements(SU2_neighbors, index, num)

        #gammas = np.append(gammas, np.asarray(unit_vec @ np.asmatrix(matrix).I))
        gammas = np.append(gammas, np.asarray(np.linalg.solve(matrix, unit_vec.T)))

        SU2_neighbors = SU2_neighbors[3:]
    
    La[element_index] = -np.sum(gammas/n)
    for index, gamma in enumerate(gammas):
        La[sort_indeces[index+1]] = gamma/n
    #should be correct
    return -1j*La


def angular_momentum(lattice_array, a: int, n=None, left=True):
    '''
    quick and dirty implementation to get the whole La matrix by calling
    'angular_momentum_single_element' for every lattice element

    use left=False to generate the Ra operator
    '''
    n = 1 if n == None else n
    La = np.zeros((lattice_array.shape[0],lattice_array.shape[0]), dtype=np.complex128)
    for index, element in enumerate(lattice_array):
        La[index,:] = angular_momentum_single_element(element, lattice_array, a, n=n, left=left)
    return La


def new_calc_alpha(element: SU2_element, neighbor: SU2_element, left=True):
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


def new_get_linear_independent(element : SU2_element, neighbors, left=True):
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
                alpha_matrix[:,j] = new_calc_alpha(element, comb[j], left=left).T
            if is_linear_independent(alpha_matrix):
                return alpha_matrix, indeces #later for more neighbors
        i += 1 
    else:
        raise BaseException("No linear independent combination passible")
    

def new_calc_gamma(alpha_matrix, a: int):
    '''
    solves e_a = gamma \dot alpha_matrix for gamma using the inverse of alpha_matrix
    '''
    unit_vec = np.zeros(3)
    unit_vec[a-1] = 1
    return np.linalg.solve(alpha_matrix, unit_vec.T)



def new_angular_momentum(lattice_array, a: int, n=None, left=True):
    '''
    new try at the implementation of the forward derivative with nx3 neighbors
    '''
    n = 1 if n == None else n
    La = np.zeros((lattice_array.shape[0], lattice_array.shape[0]))
    su2_lattice = SU2_element.vectorize_init(lattice_array)

    for index, element in enumerate(lattice_array):
        _, neighbors_indeces = get_neighbors_single_element(element, lattice_array)
        su2_neighbors = su2_lattice[neighbors_indeces]
        for neighbor_group in range(n):
            alpha_matrix, inds = new_get_linear_independent(SU2_element(element), su2_neighbors, left=left)
            gammas = new_calc_gamma(alpha_matrix, a)
            La[index, index] = -np.sum(gammas)
            for i,ind in enumerate(inds):
                La[index, neighbors_indeces[ind]] = gammas[i]
            su2_neighbors = np.delete(su2_neighbors, inds, axis=0)


    return -1j*La/n



def main():

    return

if __name__ == "__main__":
    main()