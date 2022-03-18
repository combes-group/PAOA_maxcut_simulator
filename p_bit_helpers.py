#!pip install noisyopt #install this if it is not installed on your machine.
import numpy as np
import math
import random
import time
import networkx as nx
from numpy.random import choice
import itertools
from scipy import optimize
from scipy.optimize import dual_annealing
from scipy.optimize import minimize
from noisyopt import minimizeSPSA
from collections import defaultdict
from operator import itemgetter


def two_bit_op(n,bit1,bit2,ps):
    """
    Creates a two bit stochastic matrix on bit 1 and bit 2 from an n 
    bit string. The matrix on bit 1 and 2 is of the form

    [0  0  0  0 ]
    [p1 p2 p3 p4]
    [q1 q2 q3 q4]
    [0  0  0  0 ]

    where pi + qi = 1 for i in [1,2,3,4].

    Parameters
    ----------
    n : integer
        Number of bits in string which represents the number of verticies 
        in graph.       
    bit1 : integer
        First vertex or first bit.
    bit2 : integer
    ps : ndarray
        Probablites to popluate stochastic matrix gate.

    Returns
    -------
    op : ndarray
        An array object represting a 2^n x 2^n stochastic matrix.
    """
    Bit1, Bit2 = sorted([bit1,bit2])
    # Define projectors   
    Id = np.eye(2)
    p00 = np.array([[1,0],[0,0]])
    p01 = np.array([[0,1],[0,0]])
    p10 = np.array([[0,0],[1,0]])
    p11 = np.array([[0,0],[0,1]])

    op1 = np.array([1])
    op2 = np.array([1])
    op3 = np.array([1])
    op4 = np.array([1])

    # see equation XXX 
    p1 = ps[0]*p00 + ps[2]*p01
    p2 = (1-ps[0])*p10 + (1-ps[2])*p11
    p3 = ps[1]*p00 + ps[3]*p01
    p4 = (1-ps[1])*p10 + (1-ps[3])*p11

    # construct the gate by adding identity to bits that dont involve the gate
    # if the ith bit is bit 1 apply the right half of the tensor product.
    for i in range(n):
        if i == Bit1:
            op1,op2,op3,op4 = np.kron(op1,p1),np.kron(op2,p2),np.kron(op3,p3),np.kron(op4,p4)
        elif i == Bit2:
            op1,op2,op3,op4 = np.kron(op1,p10),np.kron(op2,p00),np.kron(op3,p11),np.kron(op4,p01)
        else:
            op1,op2,op3,op4 = np.kron(op1,Id),np.kron(op2,Id),np.kron(op3,Id),np.kron(op4,Id)
    op = op1 + op2 + op3 + op4 
    return op


def get_circuit(G,ps):
    """
    Given a graph G returns a 2^n x 2^n matrix which is our variational circuit.
    For each edge in the graph the function dot products each individual 2^n x 2^n
    operator to get the total operations of the circuit.

    Parameters
    ----------
    G : networkx graph object
        Represents the graph that we are doing max cut on
    ps : ndarray
        Probablites to popluate stochastic matrix gate.

    Returns
    -------
    circuit : ndarray
        An array object represting a 2^n x 2^n total circuit.
    """
    N = G.number_of_nodes()
    circuit = np.eye(2**N)
    k = 0
    # for each edge apply a stochastic matrix gate
    for i,j in G.edges():
        op = two_bit_op(N,i,j,ps[4*k:4*k+4])
        circuit = np.dot(op,circuit)
        k += 1
    return circuit


def get_state_space(n):
    """
    Given the number of vertices n returns a two dimentional list where the first
    dimention is the set of basis vectors for the 2^n space, and the second dimention
    is the set of corresponding bit strings. 

    Parameters
    ----------
    n : integer
        Number of bits in string which represents the number of verticies 
        in graph.       

    Returns
    -------
    states_strings : list
        An 2-dimentional list object where the first dimention is the set of basis
        vectors (ndarray) for the 2^n space, and the second dimention is the set of corresponding
        bit strings (list). 
    """
    N = 2**n
    #initialize empty lists
    strings = [None]*N
    states = [None]*N
    
    # get the 2^n-by-1 basis states
    for i in range(2**n):
        v = np.zeros((2**n,1))
        v[i] = 1
        states[i] = v

    # get the length n bit strings
    lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
    for i in range(len(lst)):
        string = ''
        for j in range(len(lst[0])):
            string = string + str(lst[i][j])
        strings[i] = string
    
    # merge lists
    states_strings = [states,strings]

    return states_strings

def randvec(n):
    """
    Given the number of vertices n returns a random 2^n x 1 basis vector 

    Parameters
    ----------
    n : integer
        Number of bits in string which represents the number of verticies 
        in graph.       

    Returns
    -------
    v : ndarray
        A random 2^n x 1 basis vector 
    """
    v = np.zeros((2**n,1))
    index = random.randint(0, 2**n - 1)
    v[index] = 1
    return v

def execute(circuit,n,iterations):
    """
    Given the number of vertices n, a circuit (matrix of size 2^n x 2^n), and number of iterations,
    returns a dictionary that specifies the number of times each bit-string was the measured result
    of the circuit.

    Parameters
    ----------
    n : integer
        Number of bits in string which represents the number of verticies 
        in graph. 
        
    circuit : ndarray
        A 2^n x 2^n stochastic matrix.
        
    iterations : integer
        Number of trials in an experiment.
    
    Returns
    -------
    result : dictionary
        A dictionary that specifies the number of times each bit-string was the measured result
        of the circuit.
    """
    N = 2**n
    states = get_state_space(n)

    result = {}
    for i in range(N):
        result.update({states[1][i]:0})

    for i in range(iterations):
        # apply circuit to a random vector
        state = randvec(n)
        state = np.dot(circuit,state)
        
        # initialize weights
        w = [0]*N
        # assign weights from probabilities at each index
        for i in range(N):
            w[i] = float(state[i])
        
        # get a measurement from the weighted distribution
        measurment = random.choices(states[1],w)
        # parse through dictionary and add a count based on the measurment
        for x in result:
            if measurment[0] == x:
                result[x] += 1

    return result

def maxcut_obj(x,G):
    """
    Given a length n bit string x and graph G, calculate max-cut objective and outputs the cut
    size for that string.

    Parameters
    ----------
    x : string
        A length n bit string.
        
    G : networkx graph object
        Represents the graph that we are doing max cut on

    Returns
    -------
    cut : int
        The value of the cut for the given string
    """
    cut = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            cut -= 1
    return cut

def compute_maxcut_energy(counts, G):
    """
    Given the dictionary counts and graph G, the function outputs the average
    max-cut energy (or cut size).

    Parameters
    ----------
    counts : dictionary
        A dictionary containing all possible length n bit strings and the resulting counts for each.
        
    G : networkx graph object
        Represents the graph that we are doing max cut on

    Returns
    -------
    avg_energy : float
        Returns the average energy of the experiment.
    """    
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items(): # go through samples that we got, for each sample:
        obj_for_meas = maxcut_obj(meas, G) # compute the value
        energy += obj_for_meas*meas_count # multiply by the number of identical samples (eg 00011)
        total_counts += meas_count # add to total
    
    avg_energy = energy/total_counts
    return avg_energy #return the expectation

def get_black_box_objective(G):
    """
    Given a graph G, the function outputs the result of the experiment for
    arbitrary parameters that can be passed using an optimizer.

    Parameters
    ----------
    G : networkx graph object
        Represents the graph that we are doing max cut on

    Returns
    -------
    f : float
        Returns the parameter-dependent average energy of the experiment.
    """    
    def f(ps):
        circuit = get_circuit(G,ps) # construct circuit
        n = G.number_of_nodes()
        counts = execute(circuit,n,100) # conduct an experiment of 100 trials

        #return the energy
        return compute_maxcut_energy(counts, G)

    return f

def conduct_experiment(type_of_graph,max_size,total):
    """
    Given a graph type, a desired maximum size and total amount of experiments for 
    each size, the function returns the data for the experiments.

    Parameters
    ----------
    type_of_graph : string
        A string containing the type of graph: 'complete' or '3-regular'.
        
    max_size : integer
        An integer specifying how big of a system you want to simulate Max-Cut for.
        if type_of_graph = 'complete', max_size must be greater or equal to 3.
        if type_of_graph = '3-regular', max_size must be even and greater or equal to 4.
    
    total : integer
        An integer specifying how many experiments you will conduct for each size system.

    Returns
    -------
    all_experiments : ndarray
        Returns a 3D numpy array. Each 2D array contains data for the n = total experiments
        conducted for a single system size. Each row in the 2D array is a single experiment
        and the columns are as following: Average energy, minimum energy, approximation ratio,
        and time it took for optimization.
    """    
    if type_of_graph == 'complete':
        all_experiments = [None]*(max_size-2)
        for j in range(3,max_size+1):
            #print("j = ",j)
            results = np.zeros((total,4))
            k = 0
            for t in range(total):
                G = nx.complete_graph(j)
                n = G.number_of_edges()
                N = 4*n
                
                init_points = np.random.uniform(0,1,size=N)
                bnds = [(0,1)]*N

                obj = get_black_box_objective(G)
                t0 = time.time()
                res_sample = minimizeSPSA(obj, bounds=bnds, x0=init_points, niter=50, paired=False)
                t1 = time.time()
                results[k][0] = res_sample['fun']
                results[k][3] = t1 - t0

                optimal_theta = res_sample['x']
                circuit = get_circuit(G, optimal_theta)
                ns = G.number_of_nodes()
                counts = execute(circuit,ns,100)

                best_cut, best_solution = min([(maxcut_obj(x,G),x) for x in counts.keys()], key=itemgetter(0))
                results[k][1] =  best_cut
                results[k][2] =  results[k][0]/results[k][1]
                #print(j, "done. Ratio: ", results[k][2], " in time: ", results[k][3])
                k += 1

            #print(results)
            #print("j-3 = ",j-3)
            all_experiments[j-3] = results
            #print(all_experiments)
        
        return all_experiments
    
    if type_of_graph == '3-regular':
        all_experiments = [None]*((max_size-4)//2 + 1)
        for j in range(4,max_size+1,2):
            #print("j = ",j)
            results = np.zeros((total,4))
            k = 0
            for t in range(total):
                G = nx.generators.random_graphs.random_regular_graph(3, j, seed=None)
                n = G.number_of_edges()
                N = 4*n
                
                init_points = np.random.uniform(0,1,size=N)
                bnds = [(0,1)]*N

                obj = get_black_box_objective(G)
                t0 = time.time()
                res_sample = minimizeSPSA(obj, bounds=bnds, x0=init_points, niter=50, paired=False)
                t1 = time.time()
                results[k][0] = res_sample['fun']
                results[k][3] = t1 - t0

                optimal_theta = res_sample['x']
                circuit = get_circuit(G, optimal_theta)
                ns = G.number_of_nodes()
                counts = execute(circuit,ns,100)

                best_cut, best_solution = min([(maxcut_obj(x,G),x) for x in counts.keys()], key=itemgetter(0))
                results[k][1] =  best_cut
                results[k][2] =  results[k][0]/results[k][1]
                #print(j, "done. Ratio: ", results[k][2], " in time: ", results[k][3])
                k += 1

            #print(results)
            #print("(j-4)//2 = ",(j-4)//2)
            all_experiments[(j-4)//2] = results
            #print(all_experiments)
        
        return all_experiments        
    
    else:
        print("Error: type_of_graph must take 'complete' or '3-regular'!")


def get_data(results):
    """
    Given a list of experimental results from conduct_experiment, returns a cleaned data set of just the
    approximation ratios.

    Parameters
    ----------
    results : list
        A list containing ndarrays.      

    Returns
    -------
    data : ndarray
        A 2-dimentional numpy array where each row contains all experimental results
        for a fixed sized system. i.e: first row is result of graph with 4 vertices, 
        second row is results of graph with 5 vertices, etc.
    """
    data = np.zeros((len(results),len(results[0])))
    for i in range(len(results)):
        for j in range(len(results[0])):
            data[i][j] = results[i][j][2]
    return data

def get_stats(data):
    """
    Given a data set from get_data, the function returns an array where each row
    contains the mean and standard deviation for a fixed system size.

    Parameters
    ----------
    data : ndarray
        A 2-dimentional numpy array containing resulting approximation ratios. 
        Each row corresponds to a different system size.

    Returns
    -------
    stats : ndarray
        A 2-dimentional numpy array where each row contains the mean and standard
        deviation for a fixed sized system. i.e: first row is stats for graph with
        4 vertices, second row is stats for graph with 5 vertices, etc.
    """
    means = np.mean(data,axis=1)
    stds = np.std(data,axis=1)
    stats = np.stack((means,stds),axis = 0)

    return np.transpose(stats)


def get_plot(complete_stats,reg_stats):
    """
    Given 2 arrays including the statistics for both complete graphs and 3-regular
    graphs, the function generates a plot to visualize the results. You can add a line to
    save plot to a file.

    Parameters
    ----------
    complete_stats : ndarray
        A 2-dimentional numpy array containing means and standard deviations for
        the complete graph experiments.
        
    reg_stats : ndarray
        A 2-dimentional numpy array containing means and standard deviations for
        the 3-regular graph experiments.

    Returns
    -------
    Nothing
    
    """
    xs_complete = np.arange(3,len(complete_stats)+3)
    xs_reglar = np.arange(4,2*(len(reg_stats)+2),2)
    plt.title("Performance Comparison of QAOA to Classical Benchmark") 
    plt.axhline(y=1, color='r', linestyle='--',label='Perfect') #horizonal line at 1.0 (perfect)
    plt.errorbar(xs_complete,complete_stats[:,0],complete_stats[:,1],linestyle='-', marker='o', color='b',capsize=10, label='Complete Graph')
    plt.errorbar(xs_reglar,reg_stats[:,0],reg_stats[:,1],linestyle='-', marker='o', color='g',capsize=10, label='3-Regular Graph')
    plt.xticks(xs_complete) # make x ticks according to the complete graph
    plt.legend()
    plt.xlabel("n (number of bits)") 
    plt.ylabel(r"$\langle C \rangle/C_{min}$")
    plt.ylim([0.0, 1.02])
    #plt.savefig('ADD PATH/pbit_data.png', dpi=300)
    plt.show()
