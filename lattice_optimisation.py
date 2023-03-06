import numpy as np
from fibonacci import generate_vertices_angles
from lattice_actions import get_neighbors, get_neighbors_kdtree
import pickle
from scipy.optimize import minimize_scalar, minimize

def angles_to_cartesian(angle_array):
    '''
    takes an angle_array and returns its corresponding lattice_array in cartesian coordinates
    '''
    def single_element(arr):
        return np.asarray((np.cos(arr[0]), np.sin(arr[0])*np.cos(arr[1]), np.sin(arr[0])*np.sin(arr[1])*np.cos(arr[2]),\
            np.sin(arr[0])*np.sin(arr[1])*np.sin(arr[2])))

    lattice_array = np.apply_along_axis(single_element,1,angle_array)
    return lattice_array

def cartesian_to_angles(lattice_array):
    '''
    converts an array of cartesian coordinates to an array of spherical ones
    '''
    def single_element(arr):
        angles1 = np.arccos(arr[0])
        angles2 = np.arccos(arr[1]/np.linalg.norm(arr[1:]))
        angles3 = np.arccos(arr[-2]/np.linalg.norm(arr[-2:])) if arr[-1]>0 else 2*np.pi - np.arccos(arr[-2]/np.linalg.norm(arr[-2:]))
        return np.array([angles1, angles2, angles3])
    
    angles_array = np.apply_along_axis(single_element, 1 , lattice_array)
    return angles_array

def nabla(xi, xj):
        dpsi_i = 2*(np.sin(xi[0])*np.cos(xj[0]) - np.sin(xj[0])*np.sin(xi[1])*np.sin(xj[1])*np.cos(xi[0])*np.cos(xi[2]-xj[2])\
            - np.sin(xj[0])* np.cos(xi[0])*np.cos(xi[1])*np.cos(xj[1]))
        dtheta_i = 2*np.sin(xi[0])*np.sin(xj[0])*(np.sin(xi[1])*np.cos(xj[1]) - np.sin(xj[1])*np.cos(xi[1])*np.cos(xi[2]-xj[2]))
        dphi_i = 2*np.sin(xi[0])*np.sin(xj[0])*np.sin(xi[1])*np.sin(xj[1])*np.sin(xi[2]-xj[2])
        return np.array([dpsi_i,dtheta_i,dphi_i])

def calc_mean_distance_angles(angle_array, neighbors=None):
    '''
    returns the mean distance of each element and its nearest n neighbors,
    as well as the Vector representing the gradient 
    used for maximising
    '''
    neighbors = 1 if neighbors==None else neighbors

    def norm(xi, xj):
        s1 = -2*np.sin(xi[0])*np.sin(xj[0])*np.sin(xi[1])*np.sin(xj[1])*np.cos(xi[2] - xj[2])     
        s2 = -2*np.sin(xi[0])*np.sin(xj[0])*np.cos(xi[1])*np.cos(xj[1])
        s3 = -2*np.cos(xi[0])*np.cos(xj[0])
        s4 = 2
        return s1+s2+s3+s4

    lattice_array = angles_to_cartesian(angle_array)
    _, neighbor_indeces = get_neighbors_kdtree(lattice_array, neighbours=neighbors)
    neighbor_indeces = neighbor_indeces[:,1:]
    result = np.zeros_like(neighbor_indeces, dtype=np.float64)
    nablas = np.zeros_like(angle_array, dtype=np.float64)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j] = norm(angle_array[i], angle_array[neighbor_indeces[i,j]])
            nablas[i] += nabla(angle_array[i], angle_array[neighbor_indeces[i,j]])
            nablas[neighbor_indeces[i,j]] += nabla(angle_array[neighbor_indeces[i,j]], angle_array[i])
    nablas = nablas/angle_array.shape[0]
    return np.mean(result), nablas, np.std(result)


def lattice_optimizer(angles, eps=None, neighbors=None):
    '''
    funtion to optimize a lattice of points by maximizing the distance between the nearest 'neighbors=n' 
    neighbors of each lattice element
    '''
    angles = angles.copy()
    eps = 1e-14 if eps == None else eps
    neighbors = 1 if neighbors == None else neighbors


    mean = 0
    old_mean = 2
    old_angles = angles.copy()
    old_nabla = np.zeros_like(angles, dtype=np.float64)
    adaptive = 1
    mean_arr = []
    deviation_arr = []

    while (abs(mean-old_mean) > eps):
        old_mean = mean
        mean, nabla, standart_deviation = calc_mean_distance_angles(angles, neighbors=neighbors)
        if mean > old_mean:
            old_angles = angles
            old_nabla = nabla
            angles += adaptive*nabla
        else:
            print('overshot')   # shouldn't be triggert. if it is, choose less neighbors
                                # also: some small lattice sizes don't converge
            angles = old_angles + adaptive*old_nabla
            tmp = mean
            mean = old_mean
            old_mean = tmp
        adaptive = abs(mean-old_mean) # don't know why this works as well as it does
        mean_arr.append(mean)
        deviation_arr.append(standart_deviation)

    return angles, mean, mean_arr, deviation_arr


def calc_mean_distance_angles_simple(angle_array, std = False):
    '''
    returns the mean distance of each element and its nearest n neighbors,
    as well as the Vector representing the gradient 
    used for maximising
    '''

    def norm(xi, xj):
        s1 = -2*np.sin(xi[0])*np.sin(xj[0])*np.sin(xi[1])*np.sin(xj[1])*np.cos(xi[2] - xj[2])     
        s2 = -2*np.sin(xi[0])*np.sin(xj[0])*np.cos(xi[1])*np.cos(xj[1])
        s3 = -2*np.cos(xi[0])*np.cos(xj[0])
        s4 = 2
        return s1+s2+s3+s4

    lattice_array = angles_to_cartesian(angle_array)
    _, neighbor_indeces = get_neighbors_kdtree(lattice_array)
    neighbor_indeces = neighbor_indeces[:,1:].flatten()
    distance_nearest_neigbour = np.zeros_like(neighbor_indeces, dtype=float)
    for i, element in enumerate(angle_array):
        distance_nearest_neigbour[i] = norm(element, angle_array[neighbor_indeces[i]])
    result = np.mean(distance_nearest_neigbour)
    if std:
        return result, np.std(distance_nearest_neigbour)
    return result

def lattice_optimizer_scipy(angle_array):
    '''
    optimizer using the scipy.optimize.minimize methode
    returns the optimized lattice and it's mean distance to nearest neighbour
    '''

    def gradient(angle_array):
        angle_array = angle_array.reshape((int(angle_array.shape[0]/3), 3))
        lattice_array = angles_to_cartesian(angle_array)
        _, neighbor_indeces = get_neighbors_kdtree(lattice_array)
        neighbor_indeces = neighbor_indeces[:,1:].flatten()
        grad = np.array([nabla(element, angle_array[neighbor_indeces[i]]) for i, element in enumerate(angle_array)])/angle_array.shape[0]
        return -grad.flatten()

    def call_correct(angle_array):
        angle_array = angle_array.reshape((int(angle_array.shape[0]/3), 3))
        return -calc_mean_distance_angles_simple(angle_array)

    bounds = [(0,np.pi), (0,np.pi), (0,2*np.pi)]*angle_array.shape[0]
    opt_res = minimize(call_correct, angle_array.flatten(), bounds=bounds, jac=gradient)
    angle_result = opt_res["x"].reshape(((int(opt_res["x"].shape[0]/3), 3)))
    mean_result = -opt_res["fun"] 
    return angle_result, mean_result



def main():
    
    return

if __name__ == "__main__":
    main()