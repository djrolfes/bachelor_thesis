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




    


def main():

    return

if __name__ == "__main__":
    main()