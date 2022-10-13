import numpy as np
from pynverse import inversefunc

def generate_vertices(N):
    '''
    generates an array of points generated on a fibonaci lattice
    '''
    output = np.zeros((N,4))

    def to_cartesian(psi,theta,phi):
        return np.asarray((np.cos(psi), np.sin(psi)*np.cos(theta), np.sin(psi)*np.sin(theta)*np.cos(phi),\
            np.sin(psi)*np.sin(theta)*np.sin(phi)))

    inv_func = (lambda x: 1/np.pi * (x - 1/2*np.sin(2*x)))
    for m in range(N):
        psi_m   = inversefunc(inv_func, m/(N+1), domain=[0,np.pi])
        theta_m = np.arccos(1 - 2*((m*np.sqrt(2)) % 1))
        phi_m   = 2*np.pi*((m*np.sqrt(3))%1)

        output[m,]    = to_cartesian(psi_m,theta_m,phi_m)

    return output


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
        raise ValueError("Vertice does not have enough neighbors.")

    


def main():
    lattice = generate_vertices(4)
    distances = vertice_distances(lattice)
    neighbors = get_neighbors(lattice)

    print(lattice)

    return

if __name__ == "__main__":
    main()