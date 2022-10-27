import numpy as np
from numba import njit
from fibonacci import generate_vertices
#all things lattice actions

@njit
def vertice_distances(lattice_array, norm=None):
    '''
    creates an NxN matrix to with distances of the different points

    default norm is euclidic.(np.linalg.norm) 
    '''
    norm = np.linalg.norm if norm is None else norm

    distances = np.zeros((lattice_array.shape[0],lattice_array.shape[0]))

    for row in range(1,lattice_array.shape[0]):
        for collumn in range(row):
            distances[row,collumn] = norm(lattice_array[row]-lattice_array[collumn])

    return distances+distances.T

@njit
def get_neighbors(lattice_array, neighbors=None):
    '''
    returns the next 'neighbors=n' neighbor vertices to every given lattice vertice by eucl. distance
    in a (#vertices, n, dimension) shaped array and the array 'sort_indeces' used for sorting the distances.
    'sort_indeces' includes an #vertices x #vertices array used for sorting the vertices by distance

    default output for neighbors is all neighbors.
    '''
    neighbors = len(lattice_array)-1 if neighbors==None else neighbors
    #try:
    distances = vertice_distances(lattice_array)
    #sort_indeces = np.argsort(distances)
    sort_indeces = np.zeros_like(distances,dtype="int32")
    output = np.zeros((lattice_array.shape[0],neighbors,lattice_array.shape[1]))
    for index, distance in enumerate(distances):#sort_i in enumerate(sort_indeces):
        sort_i = np.argsort(distance)
        output[index,:,:] = lattice_array[sort_i[1:neighbors+1]]
        sort_indeces[index,::] = sort_i
    return output, sort_indeces
    #except ValueError:
    #    raise ValueError("Vertex does not have enough neighbors.")

def get_neighbors_single_element(element, lattice_array):
    '''
    calculates the nearest neighbors to a given vertex using the eucl. norm and returns the neighboring
    elements ordered and an array of indeces used for ordering.

    the element itself is included as the element with the lowest distance
    '''
    norm = np.linalg.norm
    distances = norm(lattice_array-element, axis=1)
    sort_indeces = np.argsort(distances)
    sorted_lattice = lattice_array[sort_indeces]
    return sorted_lattice, sort_indeces

def main():
    lattice = generate_vertices(2**3)
    output, indeces = get_neighbors_single_element(lattice[1],lattice)
    output2, indeces2 = get_neighbors(lattice)

    return

if __name__ == "__main__":
    main()