import numpy as np
from numba.experimental import jitclass
from numba import float64, int32
from numba import jit


class SU2_element:
    def __init__(self,params):
        '''
        initialize the SU2_element class, an element is saved in an array [a,b,c,d] containing
        [[  a+ib    c+id]
        [   -c+id   a-ib]]
        '''
        self.params = params
        self.trace = 2*params[0]


    @classmethod
    def from_matrix(cls, matrix_rep):
        '''
        classmethod used to create the Su2 element from a given 2x2 matrix 
        using SU2_element.from_matrix(matrix__rep)
        '''
        a = np.real(matrix_rep[0,0])
        b = np.imag(matrix_rep[0,0])
        c = np.real(matrix_rep[0,1])
        d = np.imag(matrix_rep[0,1])
        return cls(np.array([a,b,c,d]))

    @classmethod
    def vectorize_init(cls, arr):
        '''
        a vectorized __init__ method to initialize an array of params as SU2_elements
        '''
        return np.array([cls(vec) for vec in arr])


    @classmethod
    def from_angles(cls, angle_arr):
        '''
        creates the SU2_element from a given array of angles [psi1, psi2, psi3]
        '''
        abs_psi = np.linalg.norm(angle_arr)
        angle_arr = angle_arr/abs_psi
        a = np.cos(abs_psi/2)
        b = np.sin(abs_psi/2)*angle_arr[2]
        c = np.sin(abs_psi/2)*angle_arr[1]
        d = np.sin(abs_psi/2)*angle_arr[0]
        return cls(np.array([a,b,c,d]))

    def get_angles(self):
        '''
        a function to return the turn angles alpha from the exp(-i/2 alpha * sigma),
        with sigma being the pauli matrices sigma_1, sigma_2 and sigma_3, of a 
        given SU2_element.
        '''
        #if abs(self.params[0]+1)<1e-8:
        #    return np.array([np.pi, 0, 0])
        magnitude = np.arccos(self.params[0])
        quotient = 2. if self.params[0] == 1. else 2*magnitude/np.sin(magnitude)
        return quotient/2 *np.array([self.params[3],self.params[2],self.params[1]])

    def matrix(self):
        '''
        return the 2x2 matrix rep of the given SU2_element
        '''
        u = self.params[0] + 1j*self.params[1]
        w = self.params[2] + 1j*self.params[3]
        return np.array([[u,w],[-np.conj(w), np.conj(u)]])

    def adjoint(self):
        '''
        returns the adjoint version of the SU(2) element
        '''
        return self.from_matrix(np.conjugate(self.matrix().T))

    def inverse(self):
        '''
        alias to adjoint, returns the inverse of the given SU2_element.
        '''
        return self.adjoint()

    def left_product(self, partner):
        '''
        updates the SU2_element, by creating the left product of the element U and another SU2_element U'
        U * U'
        '''
        a = self.params[0]*partner.params[0] - self.params[1]*partner.params[1]\
             - self.params[2]*partner.params[2] - self.params[3]*partner.params[3]
        b = self.params[1]*partner.params[0] + self.params[0]*partner.params[1]\
             + self.params[2]*partner.params[3] - self.params[3]*partner.params[2]
        c = self.params[0]*partner.params[2] + self.params[2]*partner.params[0]\
             + self.params[3]*partner.params[1] - self.params[1]*partner.params[3]
        d = self.params[0]*partner.params[3] + self.params[1]*partner.params[2]\
             + self.params[3]*partner.params[0] - self.params[2]*partner.params[1]
        
        self.params = np.array([a,b,c,d])
    
    def right_product(self, partner):
        '''
        updates the SU2_element, by creating the right product of the element U and another SU2_element U'
        U' * U
        '''
        a = partner.params[0]*self.params[0] - partner.params[1]*self.params[1]\
             - partner.params[2]*self.params[2] - partner.params[3]*self.params[3]
        b = partner.params[1]*self.params[0] + partner.params[0]*self.params[1]\
             + partner.params[2]*self.params[3] - partner.params[3]*self.params[2]
        c = partner.params[0]*self.params[2] + partner.params[2]*self.params[0]\
             + partner.params[3]*self.params[1] - partner.params[1]*self.params[3]
        d = partner.params[0]*self.params[3] + partner.params[1]*self.params[2]\
             + partner.params[3]*self.params[0] - partner.params[2]*self.params[1]
        self.params = np.array([a,b,c,d])
    

    def renormalise(self):
        '''
        renormalise the SU2_element
        '''
        self.params = self.params/np.linalg.norm(self.params)


def su2_product(left_element: SU2_element, right_element: SU2_element) -> SU2_element:
    '''
    Function to create the product of two SU(2) elements in fundamental rep
    '''
    return SU2_element.from_matrix(np.dot(left_element.matrix(),right_element.matrix()))


def get_color_states(lattice, i=0, j=0):
    '''
    create the components in color space of U as a diagonal matrix
    '''
    #J = 1/np.sqrt(np.sum(lattice**2, axis=1))
    #print("J:\n", J)
    if (i==0 and j==0) or (i==1 and j==1):
        u = (lattice[:,0] + 1j*lattice[:,1])#*J
    if (i==1 and j==1):
        u = np.conjugate(u)
    if (i != j):
        u = (lattice[:,2] + 1j*lattice[:,3])#*J
    if (i==1 and j==0):
        u = -np.conjugate(u)

    return np.diag(u)




def main():

    return

if __name__ == "__main__":
    main()