import networkx as nx 
import numpy as np
import numpy.linalg as LA
import util
import cvxpy as cp
#import matplotlib.pyplot as plt
import collections
from random import Random
import sys
import pickle

if __name__ == "__main__":
    seed = 4
    np.random.seed(seed)
    n = 33
    G = util.select_graph_from_roofnet(n)
    #budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    budget = 0.1
    l = 10
    low = 0.01
    high = 0.8
    open_file = open("./cands/cands_random{}_n{}_l{}_low{}_high{}".format(seed, n, l, int(low*1000), int(high*1000)), "rb")

    #open_file = open("./cands_random{}_n{}_l{}_low{}_high{}_budget{}".format(seed, n, l, int(low*1000), int(high*1000), budget), "rb")
    cands = pickle.load(open_file)
    open_file.close()