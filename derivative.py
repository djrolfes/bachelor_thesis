import numpy as np
#from numba import njit
from itertools import combinations
from su2_element import SU2_element, su2_product
from fibonacci import generate_vertices
from lattice_actions import *


#@njit
def is_linear_independent(matrix):
    '''
    returns if the vectors making up a given matrix are linear independent, by calculating the eigenvalues
    of the matrix and checking wether one is zero. 
    '''
    eigs, _ = np.linalg.eig(matrix)
    return np.all(eigs)

#@njit
def get_linear_independent(neighbors: SU2_element):
    '''
    returns a matrix of linear angles alpha, used if the nearest three neighbors aren't independent
    '''
    i = 4
    while i<len(neighbors):
        combs = combinations(neighbors[:i], r=3)
        index_combs = np.array(list(combinations(range(1,i+1), r=3)))
        for index,comb in enumerate(combs):
            alphas = [elem.get_angles() for elem in comb]
            matrix = np.column_stack(alphas)
            if is_linear_independent(matrix) == True:
                neighbor_index_swap = np.array(range(len(neighbors)+1))
                neighbor_index_swap[1:4:] = np.array(index_combs[index])
                return matrix, neighbor_index_swap
        i += 1
    else:
        raise("No linear independent combination passible")



def forward_derivative(lattice_array, a: int, neighbors=None):
    '''
    the forward derivative as described by the write up, returns the discrete operator L_a
    '''
    if neighbors==None:
        neighbors, neighbors_indeces = get_neighbors(lattice_array)
    SU2_lattice = SU2_element.vectorize_init(lattice_array)
    La = np.zeros((len(lattice_array), len(lattice_array)))
    unit_vector = np.array([0,0,0])
    unit_vector[a-1] = 1


    for index, vertex in enumerate(SU2_lattice):
        curr_neighbors = neighbors[index]
        curr_neighbors = SU2_element.vectorize_init(curr_neighbors)

        alphas = [curr_neighbor.get_angles() for curr_neighbor in curr_neighbors[:3]]
        matrix = np.column_stack(alphas)
        if not is_linear_independent(matrix):
            # check for linear independence of the (alpha_i) matrix
            matrix,neighbor_index_swap = get_linear_independent(curr_neighbors)
            neighbors_indeces[index] = neighbors_indeces[index][neighbor_index_swap]
        else:
            curr_neighbors = curr_neighbors[:3]
        gammas = np.linalg.solve(matrix.T, unit_vector.T)
        La[index][index] = -np.sum(gammas)
        for i,gamma in enumerate(gammas):
            La[index][neighbors_indeces[index][i+1]] = gamma

    return La
        

def derivative_single_element(element, lattice_array, a: int, n=None, neighbors=None):
    '''
    calculates the derivative by using the 3n nearest neighbors of each element
    '''
    n = 1 if n == None else n
    unit_vec = np.array([0,0,0])
    unit_vec[a-1] = 1
    if neighbors == None:
        neighbors, sort_indeces = get_neighbors_single_element(element, lattice_array)
    #tbd: calc sort_indeces from given neighbors and the latticearray

    element = SU2_element(element)
    element_index = sort_indeces[0]
    SU2_neighbors = SU2_element.vectorize_init(lattice_array)[sort_indeces][1::]

    La = np.zeros_like(neighbors)
    gammas = np.array([])
    for neighbor_group in range(n):
        connecting_elements = np.array([su2_product(elem, element.inverse()) for elem in SU2_neighbors[:3]])
        alphas = [connecting_element.get_angles() for connecting_element in connecting_elements]
        matrix = np.column_stack(alphas)

        gammas = np.append(gammas, np.linalg.solve(matrix.T))
        SU2_neighbors = SU2_neighbors[3:]









def main():
    lattice = generate_vertices(2**8)
    L1 = forward_derivative(lattice, 1)
    L2 = forward_derivative(lattice, 2)
    L3 = forward_derivative(lattice, 3)
    return

if __name__ == "__main__":
    main()