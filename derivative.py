import numpy as np
from su2_element import SU2_element, su2_product
from fibonacci import generate_vertices

def vertice_distances(lattice_array):
    '''
    creates an NxN matrix to with distances of the different points
    '''
    distances = np.zeros((lattice_array.shape[0],lattice_array.shape[0]))

    for row in range(1,lattice_array.shape[0]):
        for collumn in range(row):
            distances[row,collumn] = np.linalg.norm(lattice_array[row]-lattice_array[collumn])

    return distances+distances.T


def get_neighbors(lattice_array, neighbors=3):
    '''
    returns the next 'neighbors=n' neighbor vertices to every given lattice vertice by eucl. distance
    in a (#vertices, n, dimension) shaped array
    '''
    try:
        distances = vertice_distances(lattice_array)
        sort_indeces = np.argsort(distances)
        output = np.zeros((lattice_array.shape[0],neighbors,lattice_array.shape[1]))
        for index, sort_i in enumerate(sort_indeces):
            output[index,:,:] = lattice_array[sort_i[1:neighbors+1]]
            return output
    except ValueError:
        raise ValueError("Vertex does not have enough neighbors.")


def main():
    lattice = generate_vertices(4)


    return

if __name__ == "__main__":
    main()