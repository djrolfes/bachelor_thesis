import numpy as np
import pathlib
from genericpath import exists
from su2McExecutor import executor

PATH = pathlib.Path("/mnt/f/Studium_Physik/Bachelorarbeit/code/bachelor_thesis/lattices/phase")

fib_paths = [list(PATH.absolute().glob(f"**/fib_{i}.csv")) for i in range(160)]
fib_paths = [path for paths in fib_paths for path in paths]
fib_beta_ph = []

optim_paths = [list(PATH.absolute().glob(f"**/optim_{i}.csv")) for i in range(160)]
optim_paths = [path for paths in optim_paths for path in paths]
optim_beta_ph = []

THREADS = 10
ex = executor(THREADS)
BETAS =np.linspace(0.1, 10, 100)

def run_executor(path, beta, cold=True):
    '''
    function to run the executor with a given path and beta
    '''
    ex.recordCPUData(
        # 8^4 lattice
        8,
        # coupling constants
        betas=beta,
        # partition type
        partition="--partition-list",
        # additional parameter for partition
        partitionOpt=path,
        # directory to store results in
        dataDir=f"tmp/scan",
        # cold or hot start
        cold=cold,
        # number of iterations to run (8000 is a safe value, smaller probably fine as well)
        sweeps=100
    )
    ex.runEvaluator(
        # directory which to evaluate
        f"tmp/scan",
        # where to save results
        f"tmp/scan.csv",
        # how many of the initial iterations
        # to throw away (5000 is a safe value, smaller probably fine as well)
        50)
    return

def read_scan():
    '''
    function to read and return the data given by the scan.csv file created by run_executor
    '''
    data = np.loadtxt("tmp/scan.csv", usecols=(1))
    return data

def init(path, prev = 0.2):
    '''
    Function to call run_executor for the initial two betas
    '''
    inds = np.where(BETAS <= prev)[0][-2:]
    inds = np.append(inds,[inds[-1]+i for i in range(1,THREADS-1)])
    curr_betas = BETAS[inds]
    run_executor(path, curr_betas)
    return read_scan()


def iter_beta(path, prev=0.2):
    '''
    iterates beta and returns data
    '''
    ind = np.where(BETAS <= prev)[0][-1]
    if ind+THREADS < 100:
        inds = np.array([ind+i for i in range(1,THREADS+1)])
    else:
        inds = np.array([i for i in range(ind+1,101)])

    curr_betas = BETAS[inds]
    run_executor(path, curr_betas)
    return read_scan()


def check_phase_trans(Plaq):
    '''
    calculates the current derivative of the last two plaquette values
    returns the index at which phase transition occurs or 0 if no transition
    occured
    '''
    for i, dat in enumerate(Plaq):
        if i==0:
            continue
        if i==1:
            prev_deriv = (dat-Plaq[i-1])
        curr_deriv = (dat-Plaq[i-1])
        if curr_deriv>2.5*prev_deriv:
            return i
        prev_deriv = curr_deriv

    return 0

    


def main():
    prev_beta_ph = 0.2
    Plaq = []
    for path in fib_paths:
        print(fib_beta_ph,"\n",path)
        data = init(path, prev=prev_beta_ph)
        Plaq.extend(data)
        print(Plaq)
        phase_trans = check_phase_trans(Plaq)
        if phase_trans != 0:
            prev_beta_ph = BETAS[abs(BETAS-(prev_beta_ph-0.2+phase_trans*0.1))<1e-5][0]#might be -0.1
            fib_beta_ph.append(prev_beta_ph)
            prev_beta_ph -= 0.5
            Plaq=[]
            continue

        curr_beta = prev_beta_ph+(THREADS-2)*0.1

        while curr_beta<10:
            data = iter_beta(path, prev=curr_beta)
            Plaq.extend(data)
            phase_trans = check_phase_trans(Plaq)
            if phase_trans != 0:
                prev_beta_ph = BETAS[abs(BETAS-(prev_beta_ph-0.2+phase_trans*0.1))<1e-5][0]#might be -0.1
                fib_beta_ph.append(prev_beta_ph)
                prev_beta_ph -= 0.5
                break
            curr_beta += THREADS*0.1
        else:
            fib_beta_ph.append("none")
        Plaq=[]
    return

if __name__ == "__main__":
    main()