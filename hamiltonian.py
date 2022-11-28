import numpy as np
from scipy.sparse.linalg import eigs
from su2_element import *
from derivative import angular_momentum
from fibonacci import generate_vertices

def generate_Ldagger_L(lattice_array: np.ndarray, n=None, left=None, ang=angular_momentum):
    '''
    returns L_dagger dot L (or Ra depending on left=True/false)
    '''
    n = 1 if n==None else n
    left = True if left==None else left
    L = np.zeros((lattice_array.shape[0], lattice_array.shape[0]), dtype=np.complex128)
    for a in [1,2,3]:
        La = ang(lattice_array, a, n=n, left=left)
        L += np.dot(np.conj(La.T), La)
    return L



def generate_hamiltonian(lattice_array: np.ndarray, n = None, ang=angular_momentum):
    '''
    returns the free Hamiltonian as an operator
    '''
    n = 1 if n==None else n
    L = generate_Ldagger_L(lattice_array, n=n, left=True,ang=ang)
    R = generate_Ldagger_L(lattice_array, n=n, left=False,ang=ang)
    return (L+R)/2



def calc_eigenvals(matrix: np.ndarray, k:int=None):
    '''
    calculates the smallest k eigenvalues of the hamiltonian generated by the lattice
    '''
    k = 10 if k==None else k
    eigenvals, _ = eigs(matrix, k=k, sigma=0.)
    return eigenvals

def continuum_spectrum(n: int):
    '''
    returns an array of the n smallest eigenvalues of the continuum spectrum
    '''
    l = 0
    res = []
    while len(res) < n:
        tmp = [l*(l+2) for _ in range((l + 1)**2)]
        res.extend(tmp)
        l += 1
    return res[:n]



def main():
    #lattice = generate_vertices(2**5)
    #ham = generate_hamiltonian(lattice, n=1)
    #eigen = calc_eigenvals(lattice)
    #print(ham, "\n", ham.shape, eigen)
    return

if __name__ == "__main__":
    main()