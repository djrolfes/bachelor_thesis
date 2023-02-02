import numpy as np
from numba import njit, jit
from fibonacci import generate_vertices, generate_vertices_angles
from scipy.spatial import KDTree
#all things lattice actions


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
def get_neighbors(lattice_array, norm=None):
    '''
    returns the next 'neighbors=n' neighbor vertices to every given lattice vertice by eucl. distance
    in a (#vertices, n, dimension) shaped array and the array 'sort_indeces' used for sorting the distances.
    'sort_indeces' includes an #vertices x #vertices array used for sorting the vertices by distance

    default output for neighbors is all neighbors.
    '''
    norm = np.linalg.norm if norm is None else norm
    output = np.zeros((lattice_array.shape[0],lattice_array.shape[0], 4))
    sort_indeces = np.zeros((lattice_array.shape[0],lattice_array.shape[0]),dtype="int32")
    for i,_ in enumerate(sort_indeces):
        output[i,:,:], sort_indeces[i,:] = get_neighbors_single_element(lattice_array[i], lattice_array, norm=norm)

    return output, sort_indeces

def get_neighbors_kdtree(lattice_array, neighbours = 1):
    '''
    returns the nearest neigbours to each element in the lattice, using a KD Tree
    '''
    tree = KDTree(lattice_array)
    result = tree.query(lattice_array, k=neighbours+1)
    return result

@njit
def get_neighbors_single_element(element, lattice_array, norm=None):
    '''
    calculates the nearest neighbors to a given vertex using the eucl. norm and returns the neighboring
    elements ordered and an array of indeces used for ordering.

    the element itself is included as the element with the lowest distance
    '''
    norm = np.linalg.norm if norm is None else norm
    distances = np.zeros(lattice_array.shape[0])
    for index, neigh in enumerate(lattice_array):
        distances[index] = norm(neigh-element)
    sort_indeces = np.argsort(distances)
    sorted_lattice = lattice_array[sort_indeces]
    return sorted_lattice, sort_indeces


def calc_mean_distance(lattice_array, norm=None):
    '''
    returns the mean distance between nearest elements of a lattice_array
    '''
    norm = np.linalg.norm if norm is None else norm

    dist = np.sort(vertice_distances(lattice_array, norm=norm))
    return np.mean(dist.T[1])

def main():
    lattice = generate_vertices(2**5)
    print(get_neighbors(lattice)[1])
    print(get_neighbors_kdtree(lattice, neighbours=2)[1])

    return

if __name__ == "__main__":
    main()