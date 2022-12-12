import numpy as np
from fibonacci import generate_vertices_angles
from lattice_actions import get_neighbors

def angles_to_cartesian(angle_array):
    '''
    takes an angle_array and returns its corresponding lattice_array in cartesian coordinates
    '''
    def single_element(arr):
        return np.asarray((np.cos(arr[0]), np.sin(arr[0])*np.cos(arr[1]), np.sin(arr[0])*np.sin(arr[1])*np.cos(arr[2]),\
            np.sin(arr[0])*np.sin(arr[1])*np.sin(arr[2])))

    lattice_array = np.apply_along_axis(single_element,1,angle_array)
    return lattice_array


def calc_mean_distance_angles(angle_array, neighbors=None):
    '''
    returns the mean distance of each element and its nearest neighbor,
    used for maximising
    '''
    neighbors = 1 if neighbors==None else neighbors

    def norm(xi, xj):
        s1 = -2*np.sin(xi[0])*np.sin(xj[0])*np.sin(xi[1])*np.sin(xj[1])*np.cos(xi[2] - xj[2])     
        s2 = -2*np.sin(xi[0])*np.sin(xj[0])*np.cos(xi[1])*np.cos(xj[1])
        s3 = -2*np.cos(xi[0])*np.cos(xj[0])
        s4 = 2
        return s1+s2+s3+s4

    def nabla(xi, xj):
        dpsi_i = 2*(np.sin(xi[0])*np.cos(xj[0]) - np.sin(xj[0])*np.sin(xi[1])*np.sin(xj[1])*np.cos(xi[0])*np.cos(xi[2]-xj[2])\
            - np.sin(xj[0])* np.cos(xi[0])*np.cos(xi[1])*np.cos(xj[1]))
        dtheta_i = 2*np.sin(xi[0])*np.sin(xj[0])*(np.sin(xi[1])*np.cos(xj[1]) - np.sin(xj[1])*np.cos(xi[1])*np.cos(xi[2]-xj[2]))
        dphi_i = 2*np.sin(xi[0])*np.sin(xj[0])*np.sin(xi[1])*np.sin(xj[1])*np.sin(xi[2]-xj[2])
        return np.array([dpsi_i,dtheta_i,dphi_i])

    lattice_array = angles_to_cartesian(angle_array)
    _, neighbor_indeces = get_neighbors(lattice_array)
    neighbor_indeces = neighbor_indeces[:,1:neighbors+1]
    result = np.zeros_like(neighbor_indeces, dtype=np.float64)
    nablas = np.zeros_like(angle_array, dtype=np.float64)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j] = norm(angle_array[i], angle_array[neighbor_indeces[i,j]])
            nablas[i] += nabla(angle_array[i], angle_array[neighbor_indeces[i,j]])
            nablas[neighbor_indeces[i,j]] += nabla(angle_array[neighbor_indeces[i,j]], angle_array[i])
    nablas = nablas/angle_array.shape[0]
    return np.mean(result), nablas


def lattice_optimizer(angles, eps=None, neighbors=None):
    '''
    funtion to optimize a lattice of points by maximizing the distance between the nearest 'neighbors=n' 
    neighbors of each lattice element
    '''
    eps = 1e-10 if eps == None else eps
    neighbors = 1 if neighbors == None else neighbors


    mean = 0
    old_mean = 2
    old_angles = angles.copy()
    old_nabla = np.zeros_like(angles, dtype=np.float64)
    adaptive = 1
    mean_arr = []

    while (abs(mean-old_mean) > eps):
        old_mean = mean
        mean, nabla = calc_mean_distance_angles(angles, neighbors=neighbors)
        if mean > old_mean:
            old_angles = angles
            old_nabla = nabla
            angles += adaptive*nabla
        else:
            print('overshot')   # shouldn't be triggert. if it is, choose less neighbors
                                # or more, some small lattice sizes don't converge
            angles = old_angles + adaptive*old_nabla
            tmp = mean
            mean = old_mean
            old_mean = tmp
        adaptive = abs(mean-old_mean) # don't know why this works as well as it does
        mean_arr.append(mean)

    return angles, mean, mean_arr





def main():
    angles = generate_vertices_angles(2**8)
    opti, mean, mean_arr = lattice_optimizer(angles, neighbors=3, eps=1e-10)
    print(opti, mean, mean_arr)
    return

if __name__ == "__main__":
    main()