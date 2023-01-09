from su2_element import SU2_element, su2_product
import numpy as np



# file to recreate the results from the genz writeup
class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        #Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[args]


@Memoize
def find_points(target:int, k:int):
    '''
    finds all integer partitions of target into k terms (k=4)
    '''
    if k==1:
        return np.array([target])
    x = np.zeros(k)
    x[0] = target # first element, [5,0,0,0]
    for j in range(1,target+1):
        sx = find_points(j,k-1)
        left = np.full(sx.shape[0], target-j)
        x = np.vstack((x, np.column_stack((left,sx))))#, axis =0)
    return x

def find_signs(n:int):
    '''
    provides the different combinations of signs for a n(=4) dimentional array
    '''
    i = np.array([i for i in range(2**n)])
    divisors = np.array([2**i for i in range(n-1,-1,-1)])
    x = 1 - 2*(np.broadcast_to(i, (n, len(i))).T // divisors % 2)
    return x

def gen_all_points_no_normalisation(M: int, k: int = 4):
    '''
    generates all Points in S^3 excluding normalisation
    '''
    pos_points = find_points(M, k)
    signs = find_signs(k)
    res = np.zeros((1,k), dtype=np.double)
    # own implementation to check
    #res = np.zeros((pos_points.shape[0]*signs.shape[0], k), int)
    #for i,sign in enumerate(signs):
    #    res[i*pos_points.shape[0]:(i+1)*pos_points.shape[0]] = pos_points*sign
    #res = np.unique(res, axis = 0)
    #

    #original implementation
    for point in pos_points:
        for sign in signs:
            if np.all(np.logical_or(point != 0,sign > 0)):
                res = np.vstack((res, point*sign))
    res = res[1:]
    return res


def gen_all_points(M: int, k: int = 4):
    '''
    generates all Points in S^3 including normalisation
    '''
    res = gen_all_points_no_normalisation(M, k=k)
    res = (1/np.linalg.norm(res, axis=1) * res.T).T
    return res

def compute_single_angle(neighbor, elem, left = True):
    '''
    compute_single_vi function, should yield the same as the one from the genz writeup
    '''
    elem = SU2_element(elem)
    neighbor = SU2_element(neighbor)
    if left:
        conn_element = su2_product(neighbor, elem.inverse())
    else:
        conn_element = su2_product(elem.inverse(), neighbor)
    return conn_element.get_angles()

def compute_angles(neighbors, elem, left = True):
    '''
    computes and returns angles for the connecting elements of a given list of neighbors
    '''
    res = np.zeros((neighbors.shape[0],3))
    for index, neighbor in enumerate(neighbors):
        res[index,:] = compute_single_angle(neighbor, elem, left=left)
    return res

def check_two_vec_for_lin_ind(v1, v2):
    '''
    checks two vectors on linear independece, returns True if they are linear independent
    '''
    cth = np.sum(v1*v2)/np.sqrt(np.sum(v1**2)*np.sum(v2**2))
    return (1 - abs(cth)) > 1e-9


def is_linear_independent(matrix, eps=1e-6):
    '''
    returns if the vectors making up a given matrix are linear independent, by calculating the eigenvalues
    of the matrix and checking wether one is zero. 
    '''
    return abs(np.linalg.det(matrix)) > eps

def calc_gamma(alpha_matrix, a: int):
    '''
    solves e_a = gamma \dot alpha_matrix for gamma using np.linalg.solve
    '''
    unit_vec = np.zeros(3)
    unit_vec[a-1] = 1
    print(alpha_matrix)
    res = np.linalg.solve(alpha_matrix, unit_vec.T)
    print(res)
    return res

def get_forward_neighbor(lattice, id:int, a:int, left:bool = True):
    '''
    generates the three next neighbors according to the writeup (new)
    '''
    res = {}
    res["id"] = np.array([id,id,id])
    res["alpha"] = np.full((3,3), np.pi, dtype=np.double)
    #best_values = np.array([np.pi]*3)*np.sqrt(3)

    N = lattice.shape[0]
    positive_neigh = {}
    positive_neigh["v"] = np.full((3,N), np.pi, dtype=np.double)
    positive_neigh["mod"] = np.array([3*np.pi**2]*N, dtype=np.double)
    positive_neigh["id"] = np.array([id]*N)

    count = 0
    mask = np.ones(N, dtype=bool)
    mask[id] = False
    for i in np.arange(N)[mask]:
        angles = compute_single_angle(lattice[i], lattice[id], left=left)
        if angles[a-1]>0:
            positive_neigh["v"][:,count] = angles
            positive_neigh["mod"][count] = np.sum(angles**2)
            positive_neigh["id"][count] = i
            count = count + 1
    
    found = 0
    t = np.argsort(positive_neigh["mod"],kind="heapsort")
    for i in np.argsort(positive_neigh["mod"]):
        v = positive_neigh["v"][:,i]
        if found ==  0:
            res["alpha"][:,0] = v
            res["id"][0] = positive_neigh["id"][i]
            found = found + 1

        elif found == 1:
            if check_two_vec_for_lin_ind(res["alpha"][:,0], v):
                res["alpha"][:,1] = v
                res["id"][1] = positive_neigh["id"][i]
                found = found + 1
        
        elif found == 2:
            res["alpha"][:,2] = v
            if is_linear_independent(res["alpha"]):
                res["alpha"][:,2] = v
                res["id"][2] = positive_neigh["id"][i]
                found = found+1
        else: 
            break

    return res



def gen_La_forward(lattice, a=1, left=True):
    '''
    calculate La according to the genz writeup
    '''
    N = lattice.shape[0]
    La = np.zeros((N,N))

    for i in range(N):
        g = get_forward_neighbor(lattice, i, a, left=left)
        neighbor_ids = g["id"]
        matrix = g["alpha"]


        gammas = calc_gamma(matrix, a)
        La[i,i] = - np.sum(gammas)
        for j, id in enumerate(neighbor_ids):
            La[i,id] = gammas[j]
    
    return -1j*La


def get_backward_neighbor(lattice, id:int, a:int, left:bool = True):
    '''
    generates the three next neighbors according to the writeup (new)
    '''
    res = {}
    res["id"] = np.array([id,id,id])
    res["alpha"] = np.full((3,3), np.pi)
    #best_values = np.array([np.pi]*3)*np.sqrt(3)

    N = lattice.shape[0]
    positive_neigh = {}
    positive_neigh["v"] = np.full((3,N), np.pi)
    positive_neigh["mod"] = np.array([3*np.pi**2]*N)
    positive_neigh["id"] = np.array([id]*N)

    count = 0
    mask = np.ones(N, dtype=bool)
    mask[id] = False
    for i in np.arange(N)[mask]:
        angles = compute_single_angle(lattice[i], lattice[id], left=left)
        if angles[a-1]<0:
            positive_neigh["v"][:,count] = angles
            positive_neigh["mod"][count] = np.sum(angles**2)
            positive_neigh["id"][count] = i
            count = count + 1
    
    found = 0
    t = np.argsort(positive_neigh["mod"])
    for i in np.argsort(positive_neigh["mod"]):
        v = positive_neigh["v"][:,i]
        if found ==  0:
            res["alpha"][:,0] = v
            res["id"][0] = positive_neigh["id"][i]
            found = found + 1

        elif found == 1:
            if check_two_vec_for_lin_ind(res["alpha"][:,0], v):
                res["alpha"][:,1] = v
                res["id"][1] = positive_neigh["id"][i]
                found = found + 1
        
        elif found == 2:
            res["alpha"][:,2] = v
            if is_linear_independent(res["alpha"]):
                res["alpha"][:,2] = v
                res["id"][2] = positive_neigh["id"][i]
                found = found+1
        else: 
            break

    return res


def gen_La(lattice, a=1, left=True):
    '''
    calculate La by using a forward and a backward part
    '''
    N = lattice.shape[0]
    La = np.zeros((N,N))

    for i in range(N):
        g_forward = get_forward_neighbor(lattice, i, a, left=left)
        g_backward = get_backward_neighbor(lattice, i, a, left=left)
        neighbor_ids_forward = g_forward["id"]
        neighbor_ids_backward = g_backward["id"]
        matrix_forward = g_forward["alpha"]
        matrix_backward = g_backward["alpha"]


        gammas_forward = calc_gamma(matrix_forward, a)
        gammas_backward = calc_gamma(matrix_backward, a)
        La[i,i] = - np.sum(gammas_forward)
        for j, id in enumerate(neighbor_ids_forward):
            La[i,id] = gammas_forward[j]

        La[i,i] += -np.sum(gammas_backward)
        for j, id in enumerate(neighbor_ids_backward):
            La[i,id] += gammas_backward[j]

    
    return -1j*La/2


def find_Neighbour_Ids(lattice, neighbors):
    '''
    finds the neighbors indeces according to the genz writeup version
    '1ebf6ce8e1385574a35e379bb5d887b4c939cee0'
    '''  
    g = neighbors[0]
    res = np.zeros(3, dtype=int)
    for i in range(3):
        g[i,:] = neighbors[1]*g[i,:]
        res[i] = np.where((lattice == tuple(g[i,:])).all(axis=1))[0]
    return res

def compute_Neighors_As(Point, a : int = 1):
    '''
    generates the neighbors according to the genz writeup version
    '1ebf6ce8e1385574a35e379bb5d887b4c939cee0'
    '''
    unsigned_point = abs(Point)
    Signs = np.zeros_like(Point)
    for i,p in enumerate(Point):
        Signs[i] = 1 if p == 0 else p/unsigned_point[i]
    #see if there are NANs
    pair1 = np.array([0,a])
    pair2 = np.array([0,1,2,3])[-(pair1+1)%4]

    res = np.vstack((unsigned_point,unsigned_point,unsigned_point))
    if all(unsigned_point[pair1] == 0) or all(unsigned_point[pair2] == 0):
        max_value = max(unsigned_point)
        max_value_index = np.argwhere(max_value == unsigned_point)[0][0]
        res[:,max_value_index] = res[:,max_value_index] - 1
        kk = np.arange(4)
        for neighbor in range(3):
            res[neighbor, np.delete(kk,max_value_index)[neighbor]] = res[neighbor, np.delete(kk,max_value_index)[neighbor]] + 1
    else:
        imax1 = np.argwhere(unsigned_point[pair1] == max(unsigned_point[pair1]))[0][0]
        imax2 = np.argwhere(unsigned_point[pair2] == max(unsigned_point[pair2]))[0][0]
        pair3 = np.array([pair1[imax1], pair2[imax2]])
        imax3 = np.argwhere(unsigned_point[pair3] == max(unsigned_point[pair3]))[0][0]
        for i, (pair, imax) in enumerate(zip([pair1, pair2, pair3], [imax1, imax2, imax3])):
            res[i, pair[imax]] -= 1
            res[i, np.delete(pair,imax)[0]] += 1

    return  res, Signs


def genLaNew(lattice, a = 1, left = True):
    '''
    generates the La operator according to the genz writeup version
    '1ebf6ce8e1385574a35e379bb5d887b4c939cee0'
    '''
    N = lattice.shape[0]
    La = np.zeros((N,N))
    for point in range(N):
        neighbors = compute_Neighors_As(lattice[point,:], a = a)
        neighbor_indeces = find_Neighbour_Ids(lattice, neighbors)
        angles = compute_angles(neighbors[0], lattice[point], left=left)
        angles = angles.T
        gammas = calc_gamma(angles, a = a)
        La[point, point] = -np.sum(gammas)
        for j, index in enumerate(neighbor_indeces):
            La[point, index] = gammas[j]
    return -1j*La


if __name__ == "__main__":
    points = gen_all_points_no_normalisation(5)
    La = genLaNew(points)
    print(La)

