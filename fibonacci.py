import numpy as np
from pynverse import inversefunc
from multiprocessing import Pool
from dataclasses import dataclass
CPU_CORES = 16

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

@dataclass
class Data:
    m : int
    N : int
    p : list


def generate_single_vertex_angle(mnp : Data):
    '''
    generates the vertex angles for a single combination [m, N, [irrationals]]
    '''
    m = mnp.m
    N = mnp.N
    p = mnp.p
    inv_func = (lambda x: 1/np.pi * (x - 1/2*np.sin(2*x)))
    psi_m   = inversefunc(inv_func, m/(N+1), domain=[0,np.pi])
    theta_m = np.arccos(1 - 2*((m*p[0]) % 1))
    phi_m   = 2*np.pi*((m*p[1])%1)
    return psi_m, theta_m, phi_m

def generate_vertices_angles(N, irr = None):
    '''
    generates an array of points generated on a fibonacci lattice,
    returns the angles in spherical coordinates
    '''
    irr = [np.sqrt(2), np.sqrt(3)] if irr == None else irr
    output = np.zeros((N,3))

    inv_func = (lambda x: 1/np.pi * (x - 1/2*np.sin(2*x)))
    for m in range(N):
        psi_m   = inversefunc(inv_func, m/(N+1), domain=[0,np.pi])
        theta_m = np.arccos(1 - 2*((m*irr[0]) % 1))
        phi_m   = 2*np.pi*((m*irr[1])%1)
        output[m,] = np.asarray((psi_m, theta_m, phi_m))
    
    return output

def generate_vertices_angles_mp(N, irr = None):
    '''
    generates an array of points generated on a fibonacci lattice,
    returns the angles in spherical coordinates using multiprocessing
    '''
    irr = [np.sqrt(2), np.sqrt(3)] if irr == None else irr
    pool = Pool()
    dat = [Data(m, N, irr) for m in range(N)]

    output = pool.map(generate_single_vertex_angle, dat)
    return np.array(output)


    


def main():

    return

if __name__ == "__main__":
    main()