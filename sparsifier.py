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
import operator
from pygraphml import Graph
from pygraphml import GraphMLParser

#IN THE FOLLOWING LINES OF CODES GRAPHS Gs ARE REPRESENTED BY NETWORKX OBJECTS
def getrho(edges, numberNodes):
    #mixing matrix optimization        
    numberLinks = len(edges)
    E = np.zeros((numberLinks, numberNodes))
    for i in range(numberLinks):
        E[i][edges[i][0]] = -1
        E[i][edges[i][1]] = 1
    
    I = np.eye(numberNodes)
    J = np.ones((numberNodes, numberNodes))/numberNodes
    # decision variables 
    weights = cp.Variable(numberLinks, nonneg=True)
    #weight = cp.Variable(nonneg=True)
    #weights = weight * np.ones(numberLinks)
    L = E.T @ cp.diag(weights) @ E
    s = cp.Variable(nonneg=True)
    constraints = [(I - L - J - s*I)<<0,(I - L - J + s*I)>>0]
    obj_fn = s
    problem = cp.Problem(cp.Minimize(obj_fn), constraints)
    problem.solve()
    #problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)
    #print("optimal weights:{}".format(list(weights.value)))
    rho = s.value**2
    #print("rho:{}".format(rho))
    #print("weight:{}".format(weight.value))
    return rho, list(weights.value)#list(weight.value * np.ones(numberLinks))

def getNumActiveNodes(L):
    res = 0
    for i in range(len(L)):
        if L[i][i]>0:
            res +=1
    return res

def getNumActiveEdges(L):
    G, _ = get_graph_from_laplacian(L)
    return len(get_edges(G))

def getNumMatchings(L):
    G, size = get_graph_from_laplacian(L)
    return len(getSubMatchings(G, size))

def getNumMatchingsByG(G):
    H = G.copy()
    return len(getSubMatchings(H, len(H.nodes())))

def getSubMatchings(G, size):
    """ Decompose the base graph into matchings """
    subgraphs = list()
    
    seed = 1234
    rng = Random()
    rng.seed(seed)
    
    # first try to get as many maximal matchings as possible
    for i in range(size-1):
        M1 = nx.max_weight_matching(G)
        if nx.is_perfect_matching(G, M1):
            G.remove_edges_from(list(M1))
            subgraphs.append(list(M1))
        else:
            edge_list = list(G.edges)
            rng.shuffle(edge_list)
            G.remove_edges_from(edge_list)
            G.add_edges_from(edge_list)

    # use greedy algorithm to decompose the remaining part
    rpart = decomposition(list(G.edges), size)
    for sgraph in rpart:
        subgraphs.append(sgraph)

    return subgraphs

def decomposition(graph, size):

    node_degree = [[i, 0] for i in range(size)]
    node_to_node = [[] for i in range(size)]
    node_degree_dict = collections.defaultdict(int)
    node_set = set()
    for edge in graph:
        node1, node2 = edge[0], edge[1]
        node_degree[node1][1] += 1
        node_degree[node2][1] += 1
        if node1 in node_to_node[node2] or node2 in node_to_node[node1]:
            print("Invalid input graph! Double edge! ("+str(node1) +", "+ str(node2)+")")
            exit()
        if node1 == node2:
            print("Invalid input graph! Circle! ("+str(node1) +", "+ str(node2)+")")
            exit()

        node_to_node[node1].append(node2)
        node_to_node[node2].append(node1)
        node_degree_dict[node1] += 1
        node_degree_dict[node2] += 1
        node_set.add(node1)
        node_set.add(node2)

    node_degree = sorted(node_degree, key = lambda x: x[1])
    node_degree[:] = node_degree[::-1]
    subgraphs = []
    min_num = node_degree[0][1]
    while node_set:
        subgraph = []
        for i in range(size):
            node1, node1_degree = node_degree[i]
            if node1 not in node_set:
                continue
            for j in range(i+1, size):
                node2, node2_degree = node_degree[j]
                if node2 in node_set and node2 in node_to_node[node1]:
                    subgraph.append((node1, node2))
                    node_degree[j][1] -= 1
                    node_degree[i][1] -= 1
                    node_degree_dict[node1] -= 1
                    node_degree_dict[node2] -= 1
                    node_to_node[node1].remove(node2)
                    node_to_node[node2].remove(node1)
                    node_set.remove(node1)
                    node_set.remove(node2)
                    break
        subgraphs.append(subgraph)
        for node in node_degree_dict:
            if node_degree_dict[node] > 0:
                node_set.add(node)
        node_degree = sorted(node_degree, key = lambda x: x[1])
        node_degree[:] = node_degree[::-1]
    return subgraphs

def getBandwidth(L, bw):
    res = 0
    G, _ = get_graph_from_laplacian(L)
    edges = list(G.edges())
    for e in edges:
        res += bw[e]
    return res

def getBandwidthByEdge(edges, bw):
    res = 0
    for e in edges:
        res += bw[e]
    return res

def placeMonitors(g, con):        
    nodes = []
    if con=="any":
        nodes = g.nodes()
    elif con=="one":
        for n in g.nodes():
            if len(n.edges())==1:
                nodes.append(n)
    elif con=="onetwo":
        for n in g.nodes():
            if len(n.edges())<=2:
                nodes.append(n)
    return nodes

def edgebetw(u,v):
    for e in u.edges():
        if e in v.edges():
            return e
    return -1

def getAdjNodes(node):
    nodes = []
    for e in node.edges():
        if e.node1 != node:
            nodes.append(e.node1)
        if e.node2 != node:
            nodes.append(e.node2)
    return nodes

def shortestpath(graph, cost, source, target):
    dist = {node:float("inf") for node in graph.nodes()}
    dist[source] = 0
    #print(dist)
    pre = {node:None for node in graph.nodes()}
    while dist:
        #print("dist:{}".format(dist))
        minNode = min(dist, key = lambda x : dist.get(x))
        if minNode == target:
            break
        mindist = dist.pop(minNode)
        #print("mindist:{}".format(mindist))
        if mindist == float("inf"):
            return -1
        #print("getAdjNodes(minNode):{}".format(getAdjNodes(minNode)))
        for node in getAdjNodes(minNode):
            if node not in dist:
                continue
            #e = edgebetw(minNode, node)
            #c=cost(e)
            c = 1.0
            if dist[node] > (mindist + c):
                dist[node] = mindist + c
                pre[node] = minNode
    pathByEdge = []
    current = target
    while pre[current]:
        pathByEdge.insert(0, edgebetw(pre[current], current))
        current = pre[current]
    return len(pathByEdge)


def getGraphBandwidth(gname, size, threshold=100000):
    resG = nx.Graph()
    parser = GraphMLParser()
    bandwidth = {}
    g = parser.parse("./archive/{}.graphml".format(gname))
    if gname == "Bics":
        monitors = placeMonitors(g, "onetwo")
    else:
        monitors = placeMonitors(g, "one")
    for i in range(len(monitors)):
        for j in range(i+1, len(monitors)):
            hopcount = shortestpath(g, 1.0, monitors[i], monitors[j])
            if hopcount <= threshold:
                bandwidth[(i,j)] = hopcount
                bandwidth[(j,i)] = hopcount
                resG.add_edge(i, j)
    if size != len(monitors):
        print("size:{}".format(size))
        print("len(monitors):{}".format(len(monitors)))
        print("size != len(monitors)")
        raise
    print("bandwidth:{}".format(bandwidth))
    return resG, bandwidth
    
def getrhoBudgetConstr(edges, numberNodes, budget):
    #mixing matrix optimization        
    numberLinks = len(edges)
    E = np.zeros((numberLinks, numberNodes))
    for i in range(numberLinks):
        E[i][edges[i][0]] = -1
        E[i][edges[i][1]] = 1
    
    I = np.eye(numberNodes)
    J = np.ones((numberNodes, numberNodes))/numberNodes
    # decision variables 
    weights = cp.Variable(numberLinks, nonneg=True)
    b = cp.Variable(numberLinks, boolean=True)
    L = E.T @ cp.diag(weights) @ E
    L0 = E.T @ cp.diag(b) @ E
    s = cp.Variable()
    constraints = [(I - L - J - s*I)<<0,(I - L - J + s*I)>>0, weights<=numberNodes*b, (cp.trace(L0)/2)<=budget*numberLinks]
    obj_fn = s
    problem = cp.Problem(cp.Minimize(obj_fn), constraints)
    #problem.solve()
    #problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)
    problem.solve(solver='ECOS_BB')
    #print("optimal weights:{}".format(list(weights.value)))
    rho = problem.value*problem.value
    #print("rho:{}".format(rho))
    return rho, list(weights.value)

def get_complete_undirected_graph(n):
    Kn = nx.complete_graph(n)
    return Kn

def get_edges(G):
    return list(G.edges)

def get_optimal_weight(G):
    edges = get_edges(G)
    rho, weights = getrho(edges, len(G.nodes))
    for i in range(len(edges)):
        (u,v) = edges[i]
        G.edges[u, v]["weight"] = weights[i]
    return rho
"""
#receive an integer n
#return an undirected unweighted complete graph of n vertices
def get_complete_undirected_graph(n):
    Kn = nx.complete_graph(n)
    return Kn

def get_complete_weighted_random_graph(n, w):
    Kn = nx.complete_graph(n)
    for (u,v) in Kn.edges():
        Kn.edges[u,v]['weight'] = w*(1-rd.uniform(0,1))
        #print(Kn.edges[u,v])
    return Kn
"""
#receive a graph G
#return the adjacency matrix of G as numpy array object
def get_adjacency_matrix(G):
    A = nx.adjacency_matrix(G)
    return A.toarray()

#receive a graph G
#return the Laplacian matrix of G as numpy array object
def get_laplacian_matrix(G):
    B = get_signed_incidence_matrix(G)
    w = []
    for (u,v) in list(G.edges()):
        w.append(G.edges[u,v]["weight"])
    return B @ np.diag(w) @ B.T
"""
def get_laplacian_matrix_(G):
    L = nx.laplacian_matrix(G)
    return L.toarray()
"""

def get_signed_incidence_matrix(G):
    numberNodes = len(G.nodes())
    numberLinks = len(G.edges())
    E = np.zeros((numberLinks, numberNodes))
    for i in range(numberLinks):
        E[i][list(G.edges())[i][0]] = -1
        E[i][list(G.edges())[i][1]] = 1
    return E.T
"""
#receive a graph G
#return the nxm signed Incidence matrix of G as numpy array object
def get_signed_incidence_matrix(G):
    B = nx.incidence_matrix(G, oriented=True)
    return B.toarray()

#Returns a mxm diagonal matrix with the weight of every edge at every diagonal entry 
def get_weight_diagonal_matrix(G):
    return np.diag(get_edge_weights(G))

#Returns a nxn diagonal matrix with the degree of every vertex as entry
def get_weighted_degree_matrix(G):
    n = G.number_of_nodes()
    return np.diag([G.degree(weight='weight')[i] for i in range(n)])

#receive a graph G
#returns the vectors corresponding to every edge of G (the column vectos of the signed incidence matrix)
def get_rank_one_laplacian_decomposition(G):
    B = nx.incidence_matrix(G, oriented=True)
    m = G.number_of_edges()
    Vs = [] #vectors vi
    for i in range(m):
        u = B[:,i]
        Vs.append(u.toarray())
    return Vs
"""
#Returns the pseudoinverse a given matrix
def get_pseudoinverse(A):
    return np.linalg.pinv(A)
"""
#Returns a matrix to the power of t
def get_matrix_tothe_power_of(A, t):
    return sp.linalg.fractional_matrix_power(A,t)

#Returns the Normalized Laplacian of a given graph G
def get_normalized_laplacian(G):
    L = get_laplacian_matrix(G)
    D = get_weighted_degree_matrix(G)
    Dsq = get_matrix_tothe_power_of(D,-0.5)
    return Dsq@L@Dsq
"""
#Returns the effective resistnace matrix R
def get_effective_resistance_matrix(G):
    L = get_laplacian_matrix(G)
    B = get_signed_incidence_matrix(G)
    Lplus = get_pseudoinverse(L)
    return B.transpose()@Lplus@B

#Returns the effective ressistance of every edge of G in a list
def get_effective_resistances(G):
    R = get_effective_resistance_matrix(G)
    m = R.shape[0]
    eff = []
    for i in range(m):
        eff.append(R[i,i])
    return eff

#Return a list of the edge weights of a graph G
def get_edge_weights(G):
    return [G.edges[(i,j)]['weight'] for (i,j) in list(G.edges)]

#Return an undirected weighted graph given a Laplacian matrix
def get_graph_from_laplacian(L):
    H = nx.Graph()
    n = L.shape[0]
    for i in range(n):
        for j in range(i+1,n):
            if (np.abs(L[i,j]) >= 0.000001):
                H.add_edge(i,j,weight = L[i,j])
    return H, n

def get_list_graph_from_laplacian(L):
    H = list()
    num = 0
    n = L.shape[0]
    for i in range(n):
        for j in range(i+1,n):
            if (L[i,j] != 0):
                num += 1
                H.append((i,j))
    return H, num
              
def Sparsify(G,q):
    n = len(G.nodes)
    eff = get_effective_resistances(G)
    m = len(eff)
    weights = np.array(get_edge_weights(G))
    ps = weights*eff/(n-1)
    ps = ps/ps.sum() 
    #print(sum(ps))
    S = np.zeros(m)
    for _ in range(q):
        i = np.random.choice(m, p=ps)
        S[i] += (1/(q*ps[i]))
    #print(S.shape)
    reweighted_edges = S*weights
    Sm = np.diag(reweighted_edges)
    B = get_signed_incidence_matrix(G)
    return B@Sm@B.transpose(), reweighted_edges

def get_parameter_q(e, n):
    C = 0.1
    #e = (1+(1/np.sqrt(n)))/5
    #print("Approximation factor: ",e)
    return int(9*(C**2)*n*np.log2(n)/(e**2))

def getSubMatchings_(G):
    """ Decompose the base graph into matchings """
    subgraphs = list()
    G = G.copy()
    seed = 1234
    rng = Random()
    rng.seed(seed)
    
    # first try to get as many maximal matchings as possible
    for i in range(len(G.nodes)-1):
        M1 = nx.max_weight_matching(G)
        if nx.is_perfect_matching(G, M1):
            G.remove_edges_from(list(M1))
            subgraphs.append(list(M1))
        else:
            edge_list = list(G.edges)
            rng.shuffle(edge_list)
            G.remove_edges_from(edge_list)
            G.add_edges_from(edge_list)
                
    # use greedy algorithm to decompose the remaining part
    rpart = decomposition_(len(G.nodes), list(G.edges))
    for sgraph in rpart:
        subgraphs.append(sgraph)

    return subgraphs

def decomposition_(n, edges):
    size = n
    node_degree = [[i, 0] for i in range(size)]
    node_to_node = [[] for i in range(size)]
    node_degree_dict = collections.defaultdict(int)
    node_set = set()
    for edge in edges:
        node1, node2 = edge[0], edge[1]
        node_degree[node1][1] += 1
        node_degree[node2][1] += 1
        if node1 in node_to_node[node2] or node2 in node_to_node[node1]:
            print("Invalid input graph! Double edge! ("+str(node1) +", "+ str(node2)+")")
            exit()
        if node1 == node2:
            print("Invalid input graph! Circle! ("+str(node1) +", "+ str(node2)+")")
            exit()

        node_to_node[node1].append(node2)
        node_to_node[node2].append(node1)
        node_degree_dict[node1] += 1
        node_degree_dict[node2] += 1
        node_set.add(node1)
        node_set.add(node2)

    node_degree = sorted(node_degree, key = lambda x: x[1])
    node_degree[:] = node_degree[::-1]
    subgraphs = []
    min_num = node_degree[0][1]
    while node_set:
        subgraph = []
        for i in range(size):
            node1, node1_degree = node_degree[i]
            if node1 not in node_set:
                continue
            for j in range(i+1, size):
                node2, node2_degree = node_degree[j]
                if node2 in node_set and node2 in node_to_node[node1]:
                    subgraph.append((node1, node2))
                    node_degree[j][1] -= 1
                    node_degree[i][1] -= 1
                    node_degree_dict[node1] -= 1
                    node_degree_dict[node2] -= 1
                    node_to_node[node1].remove(node2)
                    node_to_node[node2].remove(node1)
                    node_set.remove(node1)
                    node_set.remove(node2)
                    break
        subgraphs.append(subgraph)
        for node in node_degree_dict:
            if node_degree_dict[node] > 0:
                node_set.add(node)
        node_degree = sorted(node_degree, key = lambda x: x[1])
        node_degree[:] = node_degree[::-1]
    return subgraphs

def graphToLaplacian(subGraphs, n):
    L_matrices = list()
    for i, subgraph in enumerate(subGraphs):
        tmp_G = nx.Graph()
        tmp_G.add_edges_from(subgraph)
        L_matrices.append(nx.laplacian_matrix(tmp_G, list(range(n))).todense())

    return L_matrices

def set_flags(iterations, probabilities, L_matrices):
    activeTopology = list()
    for _ in range(iterations+1):
        activeTopology.append(np.zeros(L_matrices[0].shape))
    flags = list()
    for i in range(len(L_matrices)):
        activated = np.random.binomial(1, probabilities[i], iterations)
        flags.append(activated)
        for ite in range(len(activated)):
            activeTopology[ite] += activated[ite]*L_matrices[i]
    return [list(x) for x in zip(*flags)]

def getProbability(L_matrices, commBudget, costModel, size, gname, threshold, neighbors_info):
    if costModel == "bandwidth":
        g, bandwidth = getGraphBandwidth(gname, size, threshold)
        num_subBandwidth = []
        for L in L_matrices:
            num_subBandwidth.append(getBandwidth(L, bandwidth))
        num_bandwidth = getBandwidthByEdge(list(g.edges()), bandwidth)
    num_subgraphs = len(L_matrices)
    p = cp.Variable(num_subgraphs)
    L = p[0]*L_matrices[0]
    if costModel == "broadcast":
        ENode_list = []
        for node_id in range(size):
            for graph_id in range(num_subgraphs):
                if neighbors_info[graph_id][node_id] != []:
                    ENode_list.append(p[graph_id])
        ENodes = sum(ENode_list)
    if costModel == "bandwidth":
        Ebw = p[0]*num_subBandwidth[0]
    for i in range(num_subgraphs-1):
        L += p[i+1]*L_matrices[i+1]
        if costModel == "bandwidth":
            Ebw += p[i+1]*num_subBandwidth[i+1]
    eig = cp.lambda_sum_smallest(L, 2)
    sum_p = p[0]
    for i in range(num_subgraphs-1):
        sum_p += p[i+1]
    
    # cvx optimization for activation probabilities
    obj_fn = eig
    if costModel == "bandwidth":
        constraint = [Ebw <= num_bandwidth*commBudget, p>=0, p<=1]
    elif costModel == "broadcast":
        constraint = [ENodes <= size*commBudget, p>=0, p<=1]
    else:
        constraint = [sum_p <= num_subgraphs*commBudget, p>=0, p<=1]
    problem = cp.Problem(cp.Maximize(obj_fn), constraint)
    #problem.solve()
    problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)
    
    # get solution
    tmp_p = p.value
    originActivationRatio = np.zeros((num_subgraphs))
    for i, pval in enumerate(tmp_p):
        originActivationRatio[i] = np.real(float(pval))
    
    return np.maximum(np.minimum(originActivationRatio, 1),0)
 
def getSubLinks(G):
    """ Decompose the base graph into edges """
    subgraphs = list()
    for edge in list(G.edges):
        subgraphs.append([edge])
    return subgraphs

def drawer(subGraphs, size):
    connect = []
    cnt = 1
    for graph in subGraphs:
        new_connect = [[] for i in range(size)]
        for edge in graph:
            node1, node2 = edge[0], edge[1]
            if new_connect[node1] != [] or new_connect[node2] != []:
                print("invalide graph! graph: "+str(cnt))
                exit()
            new_connect[node1].append(node2)
            new_connect[node2].append(node1)
        # print(new_connect)
        connect.append(new_connect)
        cnt += 1
    return connect

def MATCHA(G, budget, costModel, size=0, gname=None, threshold=100000):
    if costModel == "time":
        subGraphs = getSubMatchings_(G)
    elif costModel == "energy" or costModel == "bandwidth" or costModel == "broadcast":
        subGraphs = getSubLinks(G)

    neighbors_info = drawer(subGraphs, size)
    L_matrices = graphToLaplacian(subGraphs, len(G.nodes))
    num_subgraphs = len(L_matrices)
    num_nodes = len(G.nodes)

    # prepare matrices
    I = np.eye(num_nodes)
    J = np.ones((num_nodes, num_nodes))/num_nodes

    mean_L = np.zeros((num_nodes,num_nodes))
    var_L = np.zeros((num_nodes,num_nodes))
    prob = getProbability(L_matrices, budget, costModel, size, gname, threshold, neighbors_info)
    print("prob:{}".format(prob))
    
    def nodeInthisMatching(node_id, matching):
        for node1, node2 in matching:
            if node_id == int(node1):
                return True
            if node_id == int(node2):
                return True
        return False
    Enodes = 0
    for node_id in range(size):
        incidentActProbs = []
        for j in range(len(subGraphs)):
            if nodeInthisMatching(node_id, subGraphs[j]):
                incidentActProbs.append(prob[j])
        nonActProb = 1
        for p in incidentActProbs:
            nonActProb = nonActProb*(1-p)
        Enodes += 1 - nonActProb
    
    for i in range(num_subgraphs):
        val = prob[i]
        mean_L += L_matrices[i]*val
        var_L += L_matrices[i]*(1-val)*val

    # SDP for mixing weight
    a = cp.Variable()
    b = cp.Variable()
    s = cp.Variable()
    obj_fn = s
    
    #constraint = [(1-s)*I - 2*a*mean_L-J + b*(np.dot(mean_L,mean_L)+2*var_L) << 0, a>=0, s>=0, b>=0, cp.square(a) <= b]
    constraint = [(1-s)*I - 2*a*mean_L-J + b*(np.dot(mean_L,mean_L)+2*var_L) << 0, (1+s)*I - 2*a*mean_L-J + b*(np.dot(mean_L,mean_L)+2*var_L) >> 0, a>=0, s>=0, b>=0, cp.square(a) <= b]
    problem = cp.Problem(cp.Minimize(obj_fn), constraint)
    #problem.solve(verbose=True)
    #problem.solve()
    
    problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)
    #print("a:{}".format(a.value))
    #print("alpha:{}".format(a.value))
    #print("rho:{}".format(s.value))
    #raise
    print("#nodes:{}, budget:{} rho:{}, Enodes:{}".format(size, budget, float(s.value), Enodes))
    return float(s.value)

def heuristic(G, budget):
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    if budget >= 1.0:
        rm = 0
    else:
        rm = int(np.ceil(num_edges - budget*num_edges))
    H = G.copy()
    edges = get_edges(G)
    for _ in range(rm):
        rho, w = getrho(edges, num_nodes)
        minweight = float("inf")
        for j in range(len(w)):
            if minweight > np.abs(w[j]):
                minweight = np.abs(w[j])
                argminweight = j
        H.remove_edge(edges[argminweight][0], edges[argminweight][1])
        del(edges[argminweight])
    rho, w = getrho(edges, num_nodes)
    B = get_signed_incidence_matrix(H)
    return rho, B@np.diag(w)@B.transpose()

def heuristic4Budgets(G, budgets):
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    rms = []
    for budget in sorted(budgets, reverse=True):
        if budget >= 1.0:
            rm = 0
        else:
            rm = np.ceil(num_edges - budget*num_edges)
        rms.append(rm)
    rms = sorted(list(set(rms)))
    print("rms:{}".format(rms))
    res_rhos = []
    res_Ls = []
    H = G.copy()
    edges = get_edges(G)
    i=0
    k=0
    while k<len(rms):
        #print("edges:{}".format(edges))
        #print("num_nodes:{}".format(num_nodes))
        
        #check condition
        if i == rms[k]:
            print("# nodes:{}, # edges:{}".format(num_nodes, len(edges)))
            rho, w = getrho(edges, num_nodes)
            B = get_signed_incidence_matrix(H.copy())
            res_rhos.append(rho)
            res_Ls.append(B@np.diag(w)@B.transpose())
            k+=1
        
        rho, w = getrho(edges, num_nodes)
        #find the min
        minweight = float("inf")
        for j in range(len(w)):
            if minweight > np.abs(w[j]):
                minweight = np.abs(w[j])
                argminweight = j
        #removal
        H.remove_edge(edges[argminweight][0], edges[argminweight][1])
        del(edges[argminweight])
        #update
        i+=1
    return res_rhos, res_Ls

def heuristic4BudgetsBandwidthCost(G, budgets, bandwidth):
    num_nodes = len(G.nodes())
    edges = get_edges(G)
    remainedBandwidth = []
    GBandwidth = getBandwidthByEdge(edges, bandwidth)
    print("GBandwidth:{}".format(GBandwidth))
    for budget in sorted(budgets, reverse=True):
        if budget >= 1.0:
            rm = GBandwidth
        else:
            rm = np.floor(budget*GBandwidth)
        remainedBandwidth.append(rm)
    remainedBandwidth = sorted(list(set(remainedBandwidth)), reverse=True)
    print("remainedBandwidth:{}".format(remainedBandwidth))
    print("len(remainedBandwidth):{}".format(len(remainedBandwidth)))
    res_rhos = []
    res_Ls = []
    H = G.copy()
    i=0
    while True:
        rho, w = getrho(edges, num_nodes)
        #find the min
        minweight = float("inf")
        for j in range(len(w)):
            if minweight > np.abs(w[j]):
                minweight = np.abs(w[j])
                argminweight = j
        #removal
        H.remove_edge(edges[argminweight][0], edges[argminweight][1])
        del(edges[argminweight])
        #check condition
        numBandwidth = getBandwidthByEdge(edges, bandwidth)
        if numBandwidth <= remainedBandwidth[i]:
            print("# nodes:{}, # edges:{}, numBandwidth:{}".format(num_nodes, len(edges), numBandwidth))
            rho, w = getrho(edges, num_nodes)
            B = get_signed_incidence_matrix(H.copy())
            res_rhos.append(rho)
            res_Ls.append(B@np.diag(w)@B.transpose())
            #check if all budgets are satisfied
            if i == len(remainedBandwidth) - 1:
                break
            i = i + 1
        while numBandwidth <= remainedBandwidth[i]:
            if i == len(remainedBandwidth) - 1:
                break
            i = i + 1
    return res_rhos, res_Ls
    
def heuristic4BudgetsMatchingCost(G, budgets):
    num_nodes = len(G.nodes())
    remainedMatchings = []
    Gmatchings = getNumMatchingsByG(G)
    print("Gmatchings:{}".format(Gmatchings))
    for budget in sorted(budgets, reverse=True):
        if budget >= 1.0:
            rm = Gmatchings
        else:
            rm = np.floor(budget*Gmatchings)
        remainedMatchings.append(rm)
    #make every element unique
    remainedMatchings = sorted(list(set(remainedMatchings)), reverse=True)
    print("remainedMatchings:{}".format(remainedMatchings))
    res_rhos = []
    res_Ls = []
    H = G.copy()
    edges = get_edges(G)
    i=0
    while True:
        rho, w = getrho(edges, num_nodes)
        #find the min
        minweight = float("inf")
        for j in range(len(w)):
            if minweight > np.abs(w[j]):
                minweight = np.abs(w[j])
                argminweight = j
        #removal
        H.remove_edge(edges[argminweight][0], edges[argminweight][1])
        del(edges[argminweight])
        #check condition
        numMatchings = getNumMatchingsByG(H)
        if numMatchings <= remainedMatchings[i]:
            print("# nodes:{}, # edges:{}, numMatchings:{}".format(num_nodes, len(edges), numMatchings))
            rho, w = getrho(edges, num_nodes)
            B = get_signed_incidence_matrix(H.copy())
            res_rhos.append(rho)
            res_Ls.append(B@np.diag(w)@B.transpose())
            #check if all budgets are satisfied
            if i == len(remainedMatchings) - 1:
                break
            i = i + 1
    return res_rhos, res_Ls

def getActNodes(G):
    nonActNodes = 0
    deg = sorted(list(G.degree), key=lambda item:item[1])
    for i, d in deg:
        if d == 0:
            nonActNodes +=1
        else:
            break
    return len(G.nodes) - nonActNodes

def heuristic4BudgetsBroadcastCost(G, budgets):
    num_nodes = len(G.nodes())
    remainedNodes = []
    GNodes = getActNodes(G)
    print("GNodes:{}".format(GNodes))
    for budget in sorted(budgets, reverse=True):
        if budget >= 1.0:
            rm = GNodes
        else:
            rm = np.ceil(budget*GNodes)
        remainedNodes.append(rm)
    #make every element unique
    remainedNodes = sorted(list(set(remainedNodes)), reverse=True)
    print("remainedNodes:{}".format(remainedNodes))
    res_rhos = []
    res_Ls = []
    H = G.copy()
    edges = get_edges(G)
    i=0
    while True:
        rho, w = getrho(edges, num_nodes)
        #find the min
        minweight = float("inf")
        for j in range(len(w)):
            if minweight > np.abs(w[j]):
                minweight = np.abs(w[j])
                argminweight = j
        #removal
        H.remove_edge(edges[argminweight][0], edges[argminweight][1])
        del(edges[argminweight])
        #check condition
        numActNodes = getActNodes(H)
        if numActNodes <= remainedNodes[i]:
            rho, w = getrho(edges, num_nodes)
            B = get_signed_incidence_matrix(H.copy())
            res_rhos.append(rho)
            L = B@np.diag(w)@B.transpose()
            res_Ls.append(L)
            eigenval, _ = np.linalg.eigh(L)
            print("From H, # nodes:{}, # edges:{}, numActNodes:{}, rho:{}, lambda_2:{}, lambda_n:{}, From edges, len(edges):{}".format(len(H.nodes()), len(H.edges()), numActNodes, rho, eigenval[1], eigenval[-1], len(edges)))
            #check if all budgets are satisfied
            if i == len(remainedNodes) - 1:
                break
            while numActNodes <= remainedNodes[i] and i < len(remainedNodes) - 1:
                i = i + 1
            if numActNodes <= remainedNodes[i] and i == len(remainedNodes) - 1:
                break
    return res_rhos, res_Ls

class graphCandidates(object):
    def __init__(self, G, l, low, high, costModel=None, budgets=None, bw=None):
        self.size = 0
        self.l = l
        self.low = low
        self.high = high
        self.G = G
        self.num_nodes = len(list(G.nodes()))
        self.L_matrices = list()
        self.numLinks = list()
        self.numMatchings = list()
        self.rhos = list()
        self.prob = list()
        self.num_bandwidth = list()
        self.costModel = costModel
        self.bw = bw
        
        #add opt to candidates
        get_optimal_weight(G)
        L = get_laplacian_matrix(G)      
        self.add(L)
        
        #add the heuristic L to the subs(candidates)
        if costModel == "time":
            rhos, Ls = heuristic4BudgetsMatchingCost(self.G, budgets)
        elif costModel == "energy":
            rhos, Ls = heuristic4Budgets(self.G, budgets)
        elif costModel == "bandwidth":
            rhos, Ls = heuristic4BudgetsBandwidthCost(self.G, budgets, bw)
        for rho, L in zip(rhos, Ls):
            self.add(L, rho)
            
        #add sparsifiers to candidates 
        for _ in range(l):
            e = np.random.uniform(low, high)
            q = get_parameter_q(e, self.num_nodes)
            L, _ = Sparsify(G,q)
            self.add(L)
            
    def add(self, L, rho=None):
        if rho is None:
            eigvals_L = LA.eigh(L)[0]
            rho = (max(1-eigvals_L[1], eigvals_L[-1]-1))**2
        _, numLink = get_list_graph_from_laplacian(L)
        if self.costModel == "bandwidth":
            self.num_bandwidth.append(getBandwidth(L, self.bw))
        self.size += 1
        self.L_matrices.append(L)
        self.numLinks.append(numLink)
        self.numMatchings.append(getNumMatchings(L))
        self.rhos.append(rho)
        return len(self.rhos) - 1 
    
    def remove(self, index):
        self.size -= 1
        del(self.L_matrices[index])
        del(self.numLinks[index])
        del(self.numMatchings[index])
        del(self.rhos[index])
        if self.costModel == "bandwidth":
            del(self.num_bandwidth[index])
        
def hybrid(cands, num_nodes, num_links, num_matchings, budget, costModel=None, num_bandwidth=None):
    print("hybrid num_links:{}".format(num_links))
    print("hybrid budget:{}".format(budget))
    #get rho and probability
    num_subgraphs = len(cands.L_matrices)

    # prepare matrices
    I = np.eye(num_nodes)
    J = np.ones((num_nodes, num_nodes))/num_nodes
    
    p = cp.Variable(num_subgraphs, nonneg=True)
    EL = p[0]*cands.L_matrices[0]
    EL2 = p[0]*(cands.L_matrices[0].T)@(cands.L_matrices[0])
    Ee = p[0]*cands.numLinks[0]
    Ematching = p[0]*cands.numMatchings[0]
    if costModel == "bandwidth":
        Ebw = p[0]*cands.num_bandwidth[0]
    sum_p = p[0]
    for i in range(num_subgraphs-1):
        EL += p[i+1]*cands.L_matrices[i+1]
        EL2 += p[i+1]*cands.L_matrices[i+1].T@cands.L_matrices[i+1]
        Ee += p[i+1]*cands.numLinks[i+1]
        Ematching += p[i+1]*cands.numMatchings[i+1]
        if costModel == "bandwidth":
            Ebw += p[i+1]*cands.num_bandwidth[i+1]
        sum_p += p[i+1]        
    
    s = cp.Variable()
    obj_fn = s
    if costModel == "time":
        print("cost model: time")
        constraint = [(I - 2*EL + EL2 - J - s*I)<<0, (I - 2*EL + EL2 - J + s*I)>>0, Ematching <= budget*num_matchings, sum_p == 1]
    elif costModel == "energy":    
        constraint = [(I - 2*EL + EL2 - J - s*I)<<0, (I - 2*EL + EL2 - J + s*I)>>0, Ee <= budget*num_links, sum_p == 1]
    elif costModel == "bandwidth":
        constraint = [(I - 2*EL + EL2 - J - s*I)<<0, (I - 2*EL + EL2 - J + s*I)>>0, Ebw <= budget*num_bandwidth, sum_p == 1]
    problem = cp.Problem(cp.Minimize(obj_fn), constraint)
    problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)
    #problem.solve()
    #normalize
    res_p = np.array(p.value)
    res_p /= res_p.sum()  
    cands.prob = list(res_p)
    
    rhos = cands.rhos.copy()
    numLinksCandidates = cands.numLinks.copy()
    prob = cands.prob.copy()
    sortedProb = sorted(enumerate(prob), key=operator.itemgetter(1), reverse=True)
    for i in range(10):
        maxi = sortedProb[i][0]
        print("i :{}, rho:{}, edges:{}, all edges:{}, prob:{}".format(maxi, rhos[maxi], numLinksCandidates[maxi], num_links, prob[maxi]))
    #print("rho -- s.value:{}".format(s.value))
    #print("rho -- p.value:{}".format(p.value))
    #print("rho -- p.value[0]:{}".format(p.value[0]))
    #print("rho -- p.value[1]:{}".format(p.value[1]))    
    return float(s.value)

def ifConnected(G):
    B = get_signed_incidence_matrix(G)
    L = B@np.identity(len(G.edges))@B.T
    if LA.eigh(L)[0][0] > 10e-10:
        print("LA.eigh(L)[0][0]:{}".format(LA.eigh(L)[0][0]))
        return False
    return True
"""
if __name__ == "__main__":
    #test out the budget
    for m in [400]:#[100, 200, 300, 400, 500, 600, 800, 1600, 3200]:
        for seed in [12]:
            print("seed:{}".format(seed))
            print("#edges:{}".format(m))
            np.random.seed(seed)
            n = 16
            #m = 1000
            #G = util.select_graph_from_roofnet(n)
            #G = util.select_graph_from_paper(3)
            #G = util.select_graph_from_paper(0)
            #G = util.select_graph_from_roofnet_sp(n, 5.5)
            #G = get_complete_undirected_graph(n)
            #budget = 0.8
            #speed = 1
            #G = nx.gnm_random_graph(n, m, seed=seed)
            G = get_complete_undirected_graph(n)
            #G = util.select_graph_from_roofnet_sp(n, speed)
            #parameter for resistance GS
            #tar = 5
            l = 5
            e = 0.8#0.01#0.5#0.005#0.3#0.002#0.3
            #raise
            C = []
            LHs = []
            print("e:{}".format(e))
    
            rho_ = get_optimal_weight(G)
            print("optmal rho:{}".format(rho_))
            #pos=nx.spring_layout(G)
            #nx.draw(G, pos, with_labels=True) #
        
            
            #L = get_laplacian_matrix(G)
            #eigvals_L = LA.eigh(L)[0]
            #print("optmal rho:{}".format(max(1-eigvals_L[1], eigvals_L[-1] - 1)**2))
            #print("Matcha Cb=1.01, rho:{}".format(MATCHA(G, 1.01)))
            
            #res = []
            #for budget in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            #    res.append(MATCHA(G, budget))
            #print("Matcha:{}".format(res))
            
            
            for _ in range(l):
                q = get_parameter_q(e, n)
                L, _ = Sparsify(G,q)
                _, numLinks = get_list_graph_from_laplacian(L)
                eigvals_L = LA.eigh(L)[0]
                rho = max(1-eigvals_L[1], eigvals_L[-1] - 1)**2
                print("rho:{}, budget:{}".format(rho, numLinks/len(list(G.edges()))))
            
       
       

if __name__ == "__main__":
    #test out the Mixed Integer SDP
    np.random.seed(5)
    n = 33
    G = util.select_graph_from_roofnet(n)
    #G = get_complete_undirected_graph(n)
    budget = 1.01

    edges = get_edges(G)
    rho, weights = getrho(edges, n)
    rhoC, weightsC = getrhoBudgetConstr(edges, n, budget)


#test out the prediction
if __name__ == "__main__":
    #compare opt, GS, MATCHA, hybrid
    n = 33
    #G = util.select_graph_from_roofnet(n)
    speed = 1#11#1##5.5#11
    G = util.select_graph_from_roofnet_sp(n, speed)
    E0 = get_edges(G)
    budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cb = 1.35*0.25
    ca = 4.7*0.8*0.25

    l = 3000
    low = 0.01#0.002#0.003
    high = 0.8#0.35#0.3
    
    printpath = "./print_roofnet{}_n{}_l{}_low{}_high{}_email_1.txt".format(speed, n, l, int(low*1000), int(high*1000))
    errorpath = "./error_roofnet{}_n{}_l{}_low{}_high{}_email_1.txt".format(speed, n, l, int(low*1000), int(high*1000))
    sys.stdout = open(printpath, 'w')
    sys.stderr = open(errorpath, "w")
    if not ifConnected(G):
        print("G is not connected")
        raise
    else:
        print("G is connected")
    

    print("#edges:{}".format(len(G.edges)))
    print("#nodes:{}".format(n))
    #create candidates
    cands = graphCandidates(G, l, low, high)
    #open_file = open("./cands_random{}_n{}_l{}_low{}_high{}_speed{}".format(seed, n, l, int(low*1000), int(high*1000), speed), "wb")
    #pickle.dump(cands, open_file)
    #open_file.close()
    if E0 != get_edges(G):
        print("G is modified")
        raise
    
    rhos = []
    costPerIters = []
    costTotal = []
    
    for budget in budgets:
        # add heuristic to candidates
        rho, L = heuristic(G, budget)
        _, numLinks = get_list_graph_from_laplacian(L)
        cands.add(L, numLinks, rho)
        if E0 != get_edges(G):
            print("G is modified")
            raise
    
    for budget in budgets:
        rho = hybrid(cands, n, len(list(G.edges())), budget)
        rhos.append(rho)
        costPerIter = 0
        for i in range(cands.size):
            costPerIter += cands.prob[i]*(n*ca + cb*2*cands.numLinks[i])
        costPerIters.append(costPerIter)
        costTotal.append((1/((1-np.sqrt(rho))**2))*costPerIter)
        if E0 != get_edges(G):
            print("G is modified")
            raise
    
    print("rhos:{}".format(rhos))
    print("costPerIters:{}".format(costPerIters))
    print("costTotal:{}".format(costTotal))
    print("opt budget:{}".format( budgets[costTotal.index(min(costTotal))] ))
    

"""

if __name__ == "__main__":
    #compare opt, GS, MATCHA, hybrid
    costModel="time"
    seed = 4#11#4#11#4#12
    np.random.seed(seed)
    n = 33#21#21#33#21#24#33#40
    #m = 400#400#600
    speed = 1#1#0#1#11#1##5.5#11
    G = util.select_graph_from_roofnet_sp(n, speed)
    gname = "rooftnet"#"rooftnet"#"Cogentco"#"GtsCe"#"DialtelecomCz"
    #threshold = 14
    bw = None
    #G, bw = getGraphBandwidth(gname, n, threshold)
    #G = nx.gnm_random_graph(n, m, seed=seed)
    E0 = get_edges(G)
    num_matchings = getNumMatchingsByG(G)
    if costModel == "bandwidth":
        num_bandwidth = getBandwidthByEdge(list(G.edges()), bw)
    else:
        num_bandwidth = 0
    budgets = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.9, 1.02]
    l = "10-3000"
    low = 0.01#0.002#0.003
    high = 0.8#0.35#0.3

    printpath = "./print_random{}_n{}_l{}_low{}_high{}_speed{}_costModel{}_gname{}_speed{}.txt".format(seed, n, l, int(low*1000), int(high*1000), speed, costModel, gname, speed)
    errorpath = "./error_random{}_n{}_l{}_low{}_high{}_speed{}_costModel{}_gname{}_speed{}.txt".format(seed, n, l, int(low*1000), int(high*1000), speed, costModel, gname, speed)
    sys.stdout = open(printpath, 'w')
    sys.stderr = open(errorpath, "w")
    if not ifConnected(G):
        print("G is not connected")
        raise
    else:
        print("G is connected")
   
    print("#edges:{}".format(len(G.edges)))
    print("#nodes:{}".format(n))
    #create candidates
    #cands_10 = graphCandidates(G, 10, low, high, costModel, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], bw)
    #cands_3000 = graphCandidates(G, 3000, low, high, costModel, [i/100 for i in range(10, 100)], bw)
    if E0 != get_edges(G):
        print("G is modified")
        raise
    #res_hybrid_10 = []
    #res_hybrid_3000 = []
    res_MATCHA = []
    res_GS = []
    for budget in budgets:
        res_MATCHA.append(MATCHA(G, budget, costModel, n, gname))
        if E0 != get_edges(G):
            print("E0:{}".format(E0))
            print("get_edges(G):{}".format(get_edges(G)))
            print("G is modified")
            raise
        #res_hybrid_10.append(hybrid(cands_10, n, len(list(G.edges())), num_matchings, budget, costModel, num_bandwidth))
        #res_hybrid_3000.append(hybrid(cands_3000, n, len(list(G.edges())), num_matchings, budget, costModel, num_bandwidth))
        if E0 != get_edges(G):
            print("G is modified")
            raise
        #print("res_hybrid_10:{}".format(res_hybrid_10))
        #print("res_hybrid_3000:{}".format(res_hybrid_3000))
        print("res_MATCHA:{}".format(res_MATCHA))
    
"""

if __name__ == "__main__":
    #try pure resist GS
    seed = 4
    np.random.seed(seed)
    n = 33
    G = util.select_graph_from_roofnet(n)
    budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    l = 3000
    low = 0.01
    high = 0.8
    
    printpath = "./print_resist_GS_random{}_n{}_l{}_low{}_high{}.txt".format(seed, n, l, int(low*1000), int(high*1000))
    errorpath = "./error_resist_GS_random{}_n{}_l{}_low{}_high{}.txt".format(seed, n, l, int(low*1000), int(high*1000))
    sys.stdout = open(printpath, 'w')
    sys.stderr = open(errorpath, "w")
    
    #create candidates
    cands = graphCandidates(G, l, low, high)
    #remove opt
    cands.remove(0)
    open_file = open("./cands_resist_GS_random{}_n{}_l{}_low{}_high{}".format(seed, n, l, int(low*1000), int(high*1000)), "wb")
    pickle.dump(cands, open_file)
    open_file.close()
    
    res_ = []
    
    for budget in budgets:
        
        res_.append(hybrid(cands, n, len(list(G.edges())), budget))

        print("res_:{}".format(res_))
        
        open_file = open("./cands_resist_GS_random{}_n{}_l{}_low{}_high{}_budget{}".format(seed, n, l, int(low*1000), int(high*1000), budget), "wb")
        pickle.dump(cands, open_file)
        open_file.close()

    
    open_file = open("./res_resist_GS_{}".format(l), "wb")
    pickle.dump(res_, open_file)
    open_file.close()


#post processing the output    
if __name__ == "__main__":
    seed = 4
    np.random.seed(seed)
    n = 33
    G = util.select_graph_from_roofnet(n)
    m = len(list(G.edges()))
    #budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    budget = 1.0
    l = 1000
    low = 0.01
    high = 0.8
    #open_file = open("./cands/cands_random{}_n{}_l{}_low{}_high{}".format(seed, n, l, int(low*1000), int(high*1000)), "rb")
    
    open_file = open("./testout/cands_resist_GS_random{}_n{}_l{}_low{}_high{}_budget{}".format(seed, n, l, int(low*1000), int(high*1000), budget), "rb")
    cands = pickle.load(open_file)
    open_file.close()
    for i in range(len(cands.rhos)):
        if cands.rhos[i] <= 0.6:
            print("i:{}, rho:{}, budget:{}, prob:{}".format(i, cands.rhos[i], cands.numLinks[i]/m, cands.prob[i]))
"""