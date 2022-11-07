import numpy as np
from itertools import product


def Linear_disc_points(m, dimensions=4):
    '''
    A function to create an array with points created via linear discretisation
    '''
    available_numbers = np.arange(1,m+2-dimensions)
    perm = np.array(list(product(available_numbers,repeat=dimensions)))
    sums = np.sum(perm,axis=1)
    indeces = [1*x==m for x in sums]#list of which permutations have sum(j_i)=m
    points = perm[indeces]
    points = np.array([points*comb for comb in (list(product([-1,1],repeat=dimensions)))])
    points = np.reshape(points, (points.shape[0]*points.shape[1], dimensions))

    result = np.asarray([point/np.linalg.norm(point) for point in points])

    return result