import numpy as np
from fibonacci import generate_vertices, generate_vertices_angles
from lattice_optimisation import lattice_optimizer, angles_to_cartesian

def simple_generate_vertices(N, seed = None):
    '''
    creates points on S_3 by using normal deviates
    '''
    rng = np.random.default_rng(seed=seed)
    Lattice = rng.normal(size=(N,4))
    normalise = 1/np.linalg.norm(Lattice, axis=1).reshape((N,1))
    Lattice = Lattice*normalise
    return Lattice



def main():
    Lattice = simple_generate_vertices(2**10)
    return

if __name__ == "__main__":
    main()