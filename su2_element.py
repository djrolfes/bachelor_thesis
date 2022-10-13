import numpy as np


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
    def generators(cls, a : int):
        '''
        returns the a = (0,1,2,3)th generator of SU(2) as an element 
        '''
        gen = np.zeros(4)
        gen[-a] = 1
        return cls(gen)


    
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
        return np.conjugate(self.matrix().T)

    def inverse(self):
        '''
        alias to adjoint, returns the inverse of the given SU2_element.
        '''
        return self.adjoint()

    

def su2_product(left_element, right_element):
    '''
    Function to create the product of two SU(2) elements in fundamental rep
    '''
    return SU2_element.from_matrix(np.dot(left_element.matrix(),right_element.matrix()))





def main():

    return

if __name__ == "__main__":
    main()