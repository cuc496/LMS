import collections
import networkx as nx
import numpy as np
import types
import cvxpy as cp
from mpi4py import MPI
from scipy.linalg import null_space
import sparsifier
import numpy.linalg as LA
from random import Random
import operator
import sys

"""
GraphProcessor

:description: GraphProcessor is designed to preprocess the communication graph,
              It specifies the activated neighbors of each node at each iteration
"""

class GraphProcessor(object):
    def __init__(self, base_graph, commBudget, rank, size, iterations, sub_type, args):
        self.rank = rank # index of worker
        self.size = size # totoal number of workers
        self.comm = MPI.COMM_WORLD
        self.commBudget = commBudget # user defined budget
        self.threshold = args.threshold

        # else: decompose the base graph
        self.base_graph = base_graph #G = nx.Graph()
        self.sub_type = sub_type
        if sub_type == "matchings":
            self.subGraphs = self.getSubMatchings() #list()
            self.L_matrices = self.graphToLaplacian()
        elif sub_type == "links":
            self.subGraphs = self.getSubLinks()
            self.L_matrices = self.graphToLaplacian()
            if args.cost == "bandwidth":
                self.num_bandwidth = list()
                _, bandwidth = sparsifier.getGraphBandwidth(args.graphname, self.size, self.threshold)
                for L in self.L_matrices:
                    self.num_bandwidth.append(sparsifier.getBandwidth(L, bandwidth))
        elif sub_type == "topologyBroadcast":
            self.subGraphs = list()
            self.subGraphs.append([])
            allEdges = []
            for e in list(self.base_graph.edges):
                allEdges.append(e)
            self.subGraphs.append(allEdges)
            self.L_matrices = self.graphToLaplacian()
        elif sub_type == "resistGS":
            self.subGraphs = list()
            self.L_matrices = list()
            self.len_edges_subG = list()
            self.rhos_subG = list()
            self.active_nodes = list()
            self.l = args.l
            self.high = args.high
            self.low = args.low
            #solve the optimal L
            sparsifier.get_optimal_weight(self.base_graph)

            #add the optimal L to the subs(candidates)
            L = sparsifier.get_laplacian_matrix(self.base_graph)
            self.active_nodes.append(sparsifier.getNumActiveNodes(L))
            self.L_matrices.append(L)
            subG, len_edges = sparsifier.get_list_graph_from_laplacian(L)
            self.subGraphs.append(subG)
            self.len_edges_subG.append(len_edges)
            eigvals_L = LA.eigh(L)[0]
            self.rhos_subG.append(max(1-eigvals_L[1], eigvals_L[-1] - 1)**2)


            #add the heuristic L to the subs(candidates)
            for b in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                rho, L = sparsifier.heuristic(self.base_graph, b)
                self.active_nodes.append(sparsifier.getNumActiveNodes(L))
                self.L_matrices.append(L)
                subG, len_edges = sparsifier.get_list_graph_from_laplacian(L)
                self.subGraphs.append(subG)
                self.len_edges_subG.append(len_edges)
                self.rhos_subG.append(rho)

            #q = sparsifier.get_parameter_q(self.e, self.size)
            for _ in range(self.l):
                e = np.random.uniform(self.low, self.high)
                q = sparsifier.get_parameter_q(e, self.size)
                LH, reweighted = sparsifier.Sparsify(self.base_graph,q)
                self.active_nodes.append(sparsifier.getNumActiveNodes(LH))
                self.L_matrices.append(LH)
                subG, len_edges = sparsifier.get_list_graph_from_laplacian(LH)
                self.subGraphs.append(subG)
                self.len_edges_subG.append(len_edges)
                eigvals_LH = LA.eigh(LH)[0]
                self.rhos_subG.append(max(1-eigvals_LH[1], eigvals_LH[-1] - 1)**2)

        elif sub_type == "allGreedyResistGS":
            self.subGraphs = list()
            self.L_matrices = list()
            self.len_edges_subG = list()
            self.rhos_subG = list()
            self.active_nodes = list()
            self.num_matchings = list()
            self.l = args.l
            self.high = args.high
            self.low = args.low
            if args.cost == "bandwidth":
                self.num_bandwidth = list()
                _, bandwidth = sparsifier.getGraphBandwidth(args.graphname, self.size, self.threshold)

            #solve the optimal L
            sparsifier.get_optimal_weight(self.base_graph)

            #add the optimal L to the subs(candidates)
            L = sparsifier.get_laplacian_matrix(self.base_graph)
            self.num_matchings.append(sparsifier.getNumMatchings(L))
            self.active_nodes.append(sparsifier.getNumActiveNodes(L))
            if args.cost == "bandwidth":
                self.num_bandwidth.append(sparsifier.getBandwidth(L, bandwidth))
            self.L_matrices.append(L)
            subG, len_edges = sparsifier.get_list_graph_from_laplacian(L)
            self.subGraphs.append(subG)
            self.len_edges_subG.append(len_edges)
            eigvals_L = LA.eigh(L)[0]
            self.rhos_subG.append(max(1-eigvals_L[1], eigvals_L[-1] - 1)**2)

            #add the heuristic L to the subs(candidates)
            budgets = [i/100 for i in range(args.heuristic_low, args.heuristic_high)]#budgets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            if args.cost == "time":
                rhos, Ls = sparsifier.heuristic4BudgetsMatchingCost(self.base_graph, budgets)
            elif args.cost == "bandwidth":
                rhos, Ls = sparsifier.heuristic4BudgetsBandwidthCost(self.base_graph, budgets, bandwidth)
            elif args.cost == "broadcast":
                rhos, Ls = sparsifier.heuristic4BudgetsBroadcastCost(self.base_graph, budgets)
            else:
                rhos, Ls = sparsifier.heuristic4Budgets(self.base_graph, budgets)
            for rho, L in zip(rhos, Ls):
                self.num_matchings.append(sparsifier.getNumMatchings(L))
                #print("self.num_matchings:{}".format(self.num_matchings))
                self.active_nodes.append(sparsifier.getNumActiveNodes(L))
                if args.cost == "bandwidth":
                    self.num_bandwidth.append(sparsifier.getBandwidth(L, bandwidth))
                self.L_matrices.append(L)
                subG, len_edges = sparsifier.get_list_graph_from_laplacian(L)
                self.subGraphs.append(subG)
                self.len_edges_subG.append(len_edges)
                self.rhos_subG.append(rho)
            
            #add sparsifiers L to the subs(candidates)
            for _ in range(self.l):
                e = np.random.uniform(self.low, self.high)
                q = sparsifier.get_parameter_q(e, self.size)
                LH, reweighted = sparsifier.Sparsify(self.base_graph,q)
                self.num_matchings.append(sparsifier.getNumMatchings(LH))
                self.active_nodes.append(sparsifier.getNumActiveNodes(LH))
                if args.cost == "bandwidth":
                    self.num_bandwidth.append(sparsifier.getBandwidth(LH, bandwidth))
                self.L_matrices.append(LH)
                subG, len_edges = sparsifier.get_list_graph_from_laplacian(LH)
                self.subGraphs.append(subG)
                self.len_edges_subG.append(len_edges)
                eigvals_LH = LA.eigh(LH)[0]
                self.rhos_subG.append(max(1-eigvals_LH[1], eigvals_LH[-1] - 1)**2)

        elif sub_type == "fixedGreedy":
            self.subGraphs = list()
            self.L_matrices = list()
            self.len_edges_subG = list()
            self.rhos_subG = list()
            self.active_nodes = list()
            self.num_matchings = list()
            self.l = args.l
            self.high = args.high
            self.low = args.low
            if args.cost == "bandwidth":
                self.num_bandwidth = list()
                _, bandwidth = sparsifier.getGraphBandwidth(args.graphname, self.size, self.threshold)

            #add the heuristic L to the subs(candidates)
            if args.cost == "time":
                rhos, Ls = sparsifier.heuristic4BudgetsMatchingCost(self.base_graph, [commBudget])
                if len(rhos) != 1:
                    print("heuristic4BudgetsMatchingCost is wrong")
                    raise
                rho = rhos[0]
                L = Ls[0]
            elif args.cost == "bandwidth":
                rhos, Ls = sparsifier.heuristic4BudgetsBandwidthCost(self.base_graph, [commBudget], bandwidth)
                if len(rhos) != 1:
                    print("heuristic4BudgetsBandwidthCost is wrong, multiple Ws")
                    raise
                rho = rhos[0]
                L = Ls[0]
            else:
                budget = commBudget
                rho, L = sparsifier.heuristic(self.base_graph, budget)
            self.num_matchings.append(sparsifier.getNumMatchings(L))
            self.active_nodes.append(sparsifier.getNumActiveNodes(L))
            if args.cost == "bandwidth":
                self.num_bandwidth.append(sparsifier.getBandwidth(L, bandwidth))
            self.L_matrices.append(L)
            subG, len_edges = sparsifier.get_list_graph_from_laplacian(L)
            self.subGraphs.append(subG)
            self.len_edges_subG.append(len_edges)
            self.rhos_subG.append(rho)

        elif sub_type == "fixedSparsifier":
            self.subGraphs = list()
            self.L_matrices = list()
            self.len_edges_subG = list()
            self.rhos_subG = list()
            self.active_nodes = list()
            self.num_matchings = list()
            self.l = args.l
            self.high = args.high
            self.low = args.low
            #get optimal
            sparsifier.get_optimal_weight(self.base_graph)

            min_rho = float("inf")
            selectedLH = False
            for _ in range(self.l):
                e = np.random.uniform(self.low, self.high)
                q = sparsifier.get_parameter_q(e, self.size)
                LH, reweighted = sparsifier.Sparsify(self.base_graph,q)

                #if it's not satisfied with the budget constraint, continue
                if args.cost == "time":
                    if sparsifier.getNumMatchings(LH)>commBudget*sparsifier.getNumMatchingsByG(self.base_graph):
                        continue
                else:
                    _, len_edges = sparsifier.get_list_graph_from_laplacian(LH)
                    if len_edges>commBudget*len(self.base_graph.edges()):
                        continue

                #if it's minimum, update
                eigvals_LH = LA.eigh(LH)[0]
                rho = max(1-eigvals_LH[1], eigvals_LH[-1] - 1)**2
                if rho<=min_rho:
                    min_rho = rho
                    selectedLH = LH

            if min_rho == float("inf"):
                print("no sparsifier satisfies the constraint")
                raise
            self.num_matchings.append(sparsifier.getNumMatchings(selectedLH))
            self.active_nodes.append(sparsifier.getNumActiveNodes(selectedLH))
            if args.cost == "bandwidth":
                self.num_bandwidth.append(sparsifier.getBandwidth(selectedLH, bandwidth))
            self.L_matrices.append(selectedLH)
            subG, len_edges = sparsifier.get_list_graph_from_laplacian(selectedLH)
            self.subGraphs.append(subG)
            self.len_edges_subG.append(len_edges)
            eigvals_LH = LA.eigh(selectedLH)[0]
            self.rhos_subG.append(max(1-eigvals_LH[1], eigvals_LH[-1] - 1)**2)
            print("# of selected L:{}".format(len(self.L_matrices)))
            print("self.rhos_subG:{}".format(self.rhos_subG))
            print("commBudget*base_matching:{}".format(commBudget*sparsifier.getNumMatchingsByG(self.base_graph)))
            print("self.num_matchings:{}".format(self.num_matchings))

        self.activeTopology = list()
        for _ in range(iterations+1):
            self.activeTopology.append(np.zeros(self.L_matrices[0].shape))
        """
        if sub_type != "resistGS" or sub_type != "allGreedyResistGS":
            self.neighbors_info = self.drawer()
        """
        #print("self.neighbors_info:{}".format(self.neighbors_info))

    def getProbability(self):
        """ compute activation probabilities for subgraphs """
        raise NotImplemented

    def getAlpha(self):
        """ compute mixing weights """
        raise NotImplemented

    def set_flags(self, iterations):
        """ generate activation flags for each iteration """
        raise NotImplemented

    def getGraphFromSub(self, subGraphs):
        G = nx.Graph()
        for edge in subGraphs:
            G.add_edges_from(edge)
        return G

    def getSubLinks(self):
        """ Decompose the base graph into edges """
        G = self.base_graph.copy()
        subgraphs = list()
        for edge in list(G.edges):
            subgraphs.append([edge])
        return subgraphs

    def getSubMatchings(self):
        """ Decompose the base graph into matchings """
        G = self.base_graph.copy()
        subgraphs = list()

        seed = 1234
        rng = Random()
        rng.seed(seed)

        # first try to get as many maximal matchings as possible
        for i in range(self.size-1):
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
        rpart = self.decomposition(list(G.edges))
        for sgraph in rpart:
            subgraphs.append(sgraph)

        return subgraphs

    def graphToLaplacian(self):
        L_matrices = list()
        for i, subgraph in enumerate(self.subGraphs):
            tmp_G = nx.Graph()
            tmp_G.add_edges_from(subgraph)
            L_matrices.append(nx.laplacian_matrix(tmp_G, list(range(self.size))).todense())

        return L_matrices


    def decomposition(self, graph):
        size = self.size

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

    def drawer(self):
        raise NotImplemented
        """
        input graph: list[list[tuples]]
                     [graph1, graph2,...]
                     graph: [edge1, edge2, ...]
                     edge: (node1, node2)
        output connect: matrix: [[]]

        if self.sub_type == "matchings" or self.sub_type == "links":
            connect = []
            cnt = 1
            for graph in self.subGraphs:
                new_connect = [[] for i in range(self.size)]
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
        elif self.sub_type == "topologyBroadcast" or self.sub_type == "resistGS" or self.sub_type == "resistGS":
            connect = []
            for graph in self.subGraphs:
                new_connect = [[] for i in range(self.size)]
                for edge in graph:
                    node1, node2 = edge[0], edge[1]
                    new_connect[node1].append(node2)
                    new_connect[node2].append(node1)
                # print(new_connect)
                connect.append(new_connect)
        return connect
        """

class FixedProcessor(GraphProcessor):
    """ wrapper for fixed communication graph """

    def __init__(self, base_graph, commBudget, rank, size, iterations, sub_type):
        super(FixedProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, sub_type)
        self.probabilities = self.getProbability()
        self.neighbor_weight = self.getAlpha()
        self.active_flags = self.set_flags(iterations + 1)
        self.neighbors_info = self.drawer()

    def getProbability(self):
        """ activation probabilities are same for subgraphs """
        return self.commBudget

    def getAlpha(self):
        """ there is an analytical expression of alpha in this case """
        L_base = np.zeros((self.size, self.size))
        for subLMatrix in self.L_matrices:
            L_base += subLMatrix
        w_b, _ = np.linalg.eig(L_base)
        lambdaList = list(sorted(w_b))
        if len(w_b) > 1:
            alpha = 2 / (lambdaList[1] + lambdaList[-1])

        return alpha

    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers
                     so that the activation flags are same

        """
        iterProb = np.random.binomial(1, self.probabilities, iterations)
        flags = list()
        idx = 0
        for prob in iterProb:
            if idx % 2 == 0:
                flags.append([0,1])
            else:
                flags.append([1,0])

            idx += 1
            # flags.append([prob for i in range(len(self.L_matrices))])

        return flags

class MatchaExactProcessor(GraphProcessor):
    def __init__(self, base_graph, commBudget, rank, size, iterations, sub_type):
        super(MatchaExactProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, sub_type)
        if sub_type != "links":
            print("sub_type should be links")
        self.probabilities = self.getProbability()
        print("probabilities:{}".format(self.probabilities))
        self.weights, self.mixingMatrixW, rho = self.getMixingMatrix()
        self.rho = [rho for _ in range(iterations + 1)]
        self.active_flags = self.set_flags(iterations + 1)
        self.neighbors_info = self.drawer()
    def getProbability(self):
        num_subgraphs = len(self.L_matrices)
        p = cp.Variable(num_subgraphs)
        L = p[0]*self.L_matrices[0]
        for i in range(num_subgraphs-1):
            L += p[i+1]*self.L_matrices[i+1]
        eig = cp.lambda_sum_smallest(L, 2)
        sum_p = p[0]
        for i in range(num_subgraphs-1):
            sum_p += p[i+1]

        # cvx optimization for activation probabilities
        obj_fn = eig
        constraint = [sum_p <= num_subgraphs*self.commBudget, p>=0, p<=1]
        problem = cp.Problem(cp.Maximize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

        # get solution
        tmp_p = p.value
        originActivationRatio = np.zeros((num_subgraphs))
        for i, pval in enumerate(tmp_p):
            originActivationRatio[i] = np.real(float(pval))

        return np.maximum(np.minimum(originActivationRatio, 1),0)

    def getMixingMatrix(self):
        def getrho(edges, numberNodes):
            #mixing matrix optimization
            numberLinks = len(edges)
            E = np.zeros((numberLinks, numberNodes))
            for i in range(numberLinks):
                E[i][edges[i][0][0]] = -1
                E[i][edges[i][0][1]] = 1

            I = np.eye(numberNodes)
            J = np.ones((numberNodes, numberNodes))/numberNodes
            # decision variables
            weights = cp.Variable(numberLinks, nonneg=True)
            L = E.T @ cp.diag(weights) @ E
            s = cp.Variable()
            constraints = [(I - L - J - s*I)<<0,(I - L - J + s*I)>>0]
            obj_fn = s
            problem = cp.Problem(cp.Minimize(obj_fn), constraints)
            #problem.solve()
            problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

            rho = problem.value*problem.value
            #print("rho:{}".format(rho))
            return rho, list(weights.value)

        edges = self.subGraphs
        rho, w = getrho(edges, self.size)

        numberLinks = len(edges)
        numberNodes = self.size

        E = np.zeros((numberLinks, numberNodes))
        for i in range(numberLinks):
            E[i][edges[i][0][0]] = -1
            E[i][edges[i][0][1]] = 1

        optL = E.T @ np.diag(w) @ E
        resW = np.identity(numberNodes) - optL
        #print("rho:{}".format(getrho(self.subGraphs, self.size)))
        #testr, testw = getrho(self.subGraphs, numberNodes)
        print("rho:{}".format(rho))
        #raise
        return w, resW, rho

    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers
                     so that the activation flags are same

        """
        flags = list()
        for i in range(len(self.L_matrices)):
            flags.append(np.random.binomial(1, self.probabilities[i], iterations))

        return [list(x) for x in zip(*flags)]

class MatchaProcessor(GraphProcessor):
    """ Wrapper for MATCHA
        At each iteration, only a random subset of subgraphs are activated
    """
    def __init__(self, base_graph, commBudget, rank, size, iterations, sub_type, args):
        super(MatchaProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, sub_type, args)
        self.rank = rank
        self.size = size
        self.costModel = args.cost
        self.gname = args.graphname
        self.neighbors_info = self.drawer()
        self.probabilities = self.getProbability()
        print("probabilities:{}".format(self.probabilities))
        self.neighbor_weight, rho = self.getAlpha()
        self.rho = [rho for _ in range(iterations + 1)]
        self.active_flags = self.set_flags(iterations + 1)


    def getProbability(self):
        print("self.costModel:{}".format(self.costModel))
        num_subgraphs = len(self.L_matrices)
        if self.costModel == "bandwidth":
            g_, bw = sparsifier.getGraphBandwidth(self.gname, self.size, self.threshold )
            if len(list(g_.nodes())) != self.size:
                print("len(list(g_.nodes())):{}".format(len(list(g_.nodes()))))
                print("self.size:{}".format(self.size))
                print("bw might be wrong")
                raise
            num_bandwidth = sparsifier.getBandwidthByEdge(list(self.base_graph.edges()), bw)
        p = cp.Variable(num_subgraphs)
        L = p[0]*self.L_matrices[0]
        if self.costModel == "broadcast":
            ENodes_list = []
            for node_id in range(self.size):
                for graph_id in range(num_subgraphs):
                    if self.neighbors_info[graph_id][node_id] != []:
                        ENodes_list.append(p[graph_id])
            ENodes = sum(ENodes_list)
        if self.costModel == "bandwidth":
            Ebw = p[0]*self.num_bandwidth[0]
        for i in range(num_subgraphs-1):
            L += p[i+1]*self.L_matrices[i+1]
            if self.costModel == "bandwidth":
                Ebw += p[i+1]*self.num_bandwidth[i+1]
        eig = cp.lambda_sum_smallest(L, 2)
        sum_p = p[0]
        for i in range(num_subgraphs-1):
            sum_p += p[i+1]

        # cvx optimization for activation probabilities
        obj_fn = eig
        if self.costModel == "bandwidth":
            constraint = [Ebw <= num_bandwidth*self.commBudget, p>=0, p<=1]
        elif self.costModel == "broadcast":
            constraint = [ENodes <= self.size*self.commBudget, p>=0, p<=1]
        else:
            constraint = [sum_p <= num_subgraphs*self.commBudget, p>=0, p<=1]
        problem = cp.Problem(cp.Maximize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

        # get solution
        tmp_p = p.value
        originActivationRatio = np.zeros((num_subgraphs))
        for i, pval in enumerate(tmp_p):
            originActivationRatio[i] = np.real(float(pval))

        return np.maximum(np.minimum(originActivationRatio, 1),0)

    def getAlpha(self):
        num_subgraphs = len(self.L_matrices)
        num_nodes = self.size

        # prepare matrices
        I = np.eye(num_nodes)
        J = np.ones((num_nodes, num_nodes))/num_nodes

        mean_L = np.zeros((num_nodes,num_nodes))
        var_L = np.zeros((num_nodes,num_nodes))
        for i in range(num_subgraphs):
            val = self.probabilities[i]
            mean_L += self.L_matrices[i]*val
            var_L += self.L_matrices[i]*(1-val)*val

        # SDP for mixing weight
        a = cp.Variable()
        b = cp.Variable()
        s = cp.Variable()
        obj_fn = s
        #constraint = [(1-s)*I - 2*a*mean_L-J + b*(np.dot(mean_L,mean_L)+2*var_L) << 0, a>=0, s>=0, b>=0, cp.square(a) <= b]
        constraint = [(1-s)*I - 2*a*mean_L-J + b*(np.dot(mean_L,mean_L)+2*var_L) << 0, (1+s)*I - 2*a*mean_L-J + b*(np.dot(mean_L,mean_L)+2*var_L) >> 0, a>=0, s>=0, b>=0, cp.square(a) <= b]
        problem = cp.Problem(cp.Minimize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)
        print("alpha:{}".format(a.value))
        print("rho:{}".format(s.value))
        #raise
        return  float(a.value), float(s.value)

    def drawer(self):
        """
        input graph: list[list[tuples]]
                     [graph1, graph2,...]
                     graph: [edge1, edge2, ...]
                     edge: (node1, node2)
        output connect: matrix: [[]]
        """
        connect = []
        cnt = 1
        for graph in self.subGraphs:
            new_connect = [[] for i in range(self.size)]
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

    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers
                     so that the activation flags are same
        """
        flags = list()
        for i in range(len(self.L_matrices)):
            activated = np.random.binomial(1, self.probabilities[i], iterations)
            flags.append(activated)
            for ite in range(len(activated)):
                self.activeTopology[ite] += activated[ite]*self.L_matrices[i]
        return [list(x) for x in zip(*flags)]

class EnergeProcessor(GraphProcessor):
    """ The only difference from MATCHA is the cost constraint, which is a function of ca, cb, commBudget
    """

    def __init__(self, base_graph, commBudget, rank, size, iterations, sub_type, ca, cb):
        super(EnergeProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, sub_type)
        if sub_type != "links":
            print("sub_type should be links, since the constraint in getProbability considers links")
        self.ca = ca
        self.cb = cb
        self.probabilities = self.getProbability()
        print("probabilities:{}".format(self.probabilities))
        self.neighbor_weight, rho = self.getAlpha()
        self.rho = [rho for _ in range(iterations+1)]
        self.active_flags = self.set_flags(iterations + 1)
        self.neighbors_info = self.drawer()
    def getProbability(self):
        num_subgraphs = len(self.L_matrices)
        num_workers = self.size
        p = cp.Variable(num_subgraphs)
        L = p[0]*self.L_matrices[0]
        for i in range(num_subgraphs-1):
            L += p[i+1]*self.L_matrices[i+1]
        eig = cp.lambda_sum_smallest(L, 2)
        # cvx optimization for activation probabilities
        obj_fn = eig
        constraint = [p>=0, p<=1]
        init = True
        for worker_id in range(num_workers):
            for graph_id in range(num_subgraphs):
                if self.neighbors_info[graph_id][worker_id] != -1:
                    if init:
                        sum_p = p[graph_id]
                        init = False
                    else:
                        sum_p += p[graph_id]
        constraint.append(sum_p <= ((self.commBudget - num_workers*self.ca)/self.cb))
        problem = cp.Problem(cp.Maximize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

        # get solution
        tmp_p = p.value
        #print("p.value:{}".format(p.value))
        originActivationRatio = np.zeros((num_subgraphs))
        for i, pval in enumerate(tmp_p):
            originActivationRatio[i] = np.real(float(pval))

        return np.minimum(originActivationRatio, 1)

    def getAlpha(self):
        num_subgraphs = len(self.L_matrices)
        num_nodes = self.size

        # prepare matrices
        I = np.eye(num_nodes)
        J = np.ones((num_nodes, num_nodes))/num_nodes

        mean_L = np.zeros((num_nodes,num_nodes))
        var_L = np.zeros((num_nodes,num_nodes))
        for i in range(num_subgraphs):
            val = self.probabilities[i]
            mean_L += self.L_matrices[i]*val
            var_L += self.L_matrices[i]*(1-val)*val

        # SDP for mixing weight
        a = cp.Variable()
        b = cp.Variable()
        s = cp.Variable()
        obj_fn = s
        constraint = [(1-s)*I - 2*a*mean_L-J + b*(np.dot(mean_L,mean_L)+2*var_L) << 0, a>=0, s>=0, b>=0, cp.square(a) <= b]
        problem = cp.Problem(cp.Minimize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)
        return  float(a.value), float(s.value)


    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers
                     so that the activation flags are same
        """
        flags = list()
        for i in range(len(self.L_matrices)):
            flags.append(np.random.binomial(1, self.probabilities[i], iterations))
        return [list(x) for x in zip(*flags)]

class BroadCastTopologyProcessor(GraphProcessor):
    def __init__(self, base_graph, commBudget, rank, size, iterations, sub_type, ca, cb):
        super(BroadCastTopologyProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, sub_type)
        if sub_type != "topologyBroadcast":
            print("sub_type should be topologyBroadcast")
        self.ca = ca
        self.cb = cb
        self.probabilities = self.getProbability()
        print("probabilities:{}".format(self.probabilities))
        self.neighbor_weight, rho = self.getAlpha()
        self.rho = [rho for _ in range(iterations+1)]
        self.active_flags = self.set_flags(iterations + 1)
        self.neighbors_info = self.drawer()
        print(self.active_flags[:20])
        #raise
    def getProbability(self):
        num_subgraphs = len(self.L_matrices)
        num_workers = self.size
        p0 = cp.Variable(nonneg=True)
        p1 = cp.Variable(nonneg=True)
        eig = cp.lambda_sum_smallest((p0*self.L_matrices[0] + p1*self.L_matrices[1]), 2)
        # cvx optimization for activation probabilities
        obj_fn = eig
        constraint = [p0+p1==1, p0*num_workers*self.ca + p1*(num_workers*(self.ca + self.cb))<= (self.commBudget)]
        problem = cp.Problem(cp.Maximize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

        return np.minimum([p0.value, p1.value], 1)

    def getAlpha(self):
        num_subgraphs = len(self.L_matrices)
        num_nodes = self.size
        # prepare matrices
        I = np.eye(num_nodes)
        J = np.ones((num_nodes, num_nodes))/num_nodes

        # SDP for mixing weight
        a = cp.Variable(nonneg=True)
        b = cp.Variable(nonneg=True)
        s = cp.Variable(nonneg=True)
        obj_fn = s
        constraint = [(1-s)*I - 2*a*self.probabilities[1]*self.L_matrices[1] -J + b*self.probabilities[1]*(self.L_matrices[1].T @ self.L_matrices[1]) << 0, cp.square(a) <= b]
        problem = cp.Problem(cp.Minimize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)
        return  float(a.value), float(s.value)


    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers
                     so that the activation flags are same
        """
        flags = list()
        g0 = np.random.binomial(1, self.probabilities[0], iterations)
        flags.append(g0)
        g1 = []
        for i in range(len(g0)):
            if g0[i]==0:
                g1.append(1)
            else:
                g1.append(0)
        flags.append(g1)
        return [list(x) for x in zip(*flags)]



class TopologySamplingPredictionProcessor(GraphProcessor):
    def __init__(self, base_graph, commBudget, rank, size, iterations, sub_type, args, num_batches):
        super(TopologySamplingPredictionProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, sub_type, args)
        if sub_type != "allGreedyResistGS" and sub_type != "fixedGreedy" and sub_type != "fixedSparsifier" :
            print("sub_type should be resistGS")
            raise
        self.ifBroadcast = args.broadcast
        self.seq = args.seq
        self.epoch = args.epoch
        self.num_batches = num_batches
        self.costModel = args.cost
        self.gname = args.graphname
        self.approach = args.approach
        rhos = []
        costPerIters = []
        costTotals = []
        minCostTotal = float("inf")
        
        #Algorithm III LMS line7-lin10
        budgets = [i/100 for i in range(args.bStart, args.bEnd, args.bInterval)]
        print("try budgets:{}".format(budgets))
        print("self.costModel:{}".format(self.costModel))
        for budget in budgets:
            prob, rho = self.getProbability(budget)
            rhos.append(rho)
            costPerIter = 0
            for i in range(len(prob)):
                if self.costModel == "time":
                    costPerIter += prob[i]*(self.size*args.ca + args.cb*self.num_matchings[i])
                elif self.costModel == "bandwidth":
                    costPerIter += prob[i]*(2*self.num_bandwidth[i])
                elif self.costModel == "broadcast":
                    costPerIter += prob[i]*(self.size*args.ca + args.cb*self.active_nodes[i])
                else:
                    costPerIter += prob[i]*(self.size*args.ca + args.cb*2*self.len_edges_subG[i])
            costPerIters.append(costPerIter)
            costTotal = (1/((1-np.sqrt(rho))**2))*costPerIter
            costTotals.append(costTotal)
            if costTotal <= minCostTotal:
                minCostTotal = costTotal
                self.probabilities = prob
        print("rhos:{}".format(rhos))
        print("costPerIters:{}".format(costPerIters))
        print("costTotals:{}".format(costTotals))
        print("minCostTotal:{}".format(minCostTotal))
        self.active_flags, self.rho  = self.set_flags(iterations + 1)
        self.mixingMatrixWs = self.getMixingMatrixs()
    
    #Algorithm III LMS line8
    def getProbability(self, budget):
        print("budget:{}".format(budget))
        num_subgraphs = len(self.L_matrices)
        num_links = len(list(self.base_graph.edges()))
        num_nodes = self.size
        num_matchings = sparsifier.getNumMatchingsByG(self.base_graph)
        if self.costModel == "bandwidth":
            _, bw = sparsifier.getGraphBandwidth(self.gname, self.size, self.threshold)
            num_bandwidth = sparsifier.getBandwidthByEdge(list(self.base_graph.edges()), bw)
        # prepare matrices
        I = np.eye(num_nodes)
        J = np.ones((num_nodes, num_nodes))/num_nodes

        p = cp.Variable(num_subgraphs, nonneg=True)
        EL = p[0]*self.L_matrices[0]
        EL2 = p[0]*self.L_matrices[0].T@self.L_matrices[0]
        Ee = p[0]*self.len_edges_subG[0]
        En = p[0]*self.active_nodes[0]
        Ematchings = p[0]*self.num_matchings[0]
        if self.costModel == "bandwidth":
            Ebw = p[0]*self.num_bandwidth[0]
        sum_p = p[0]
        for i in range(num_subgraphs-1):
            EL += p[i+1]*self.L_matrices[i+1]
            EL2 += p[i+1]*self.L_matrices[i+1].T@self.L_matrices[i+1]
            Ee += p[i+1]*self.len_edges_subG[i+1]
            En += p[i+1]*self.active_nodes[i+1]
            Ematchings += p[i+1]*self.num_matchings[i+1]
            if self.costModel == "bandwidth":
                Ebw += p[i+1]*self.num_bandwidth[i+1]
            sum_p += p[i+1]

        s = cp.Variable()
        obj_fn = s
        if self.costModel == "time":
            print("matching cost model")
            constraint = [(I - 2*EL + EL2 - J - s*I)<<0, (I - 2*EL + EL2 - J + s*I)>>0, Ematchings <= budget*num_matchings, sum_p == 1]
        elif self.costModel == "bandwidth":
            print("bandwidth cost model")
            constraint = [(I - 2*EL + EL2 - J - s*I)<<0, (I - 2*EL + EL2 - J + s*I)>>0, Ebw <= budget*num_bandwidth, sum_p == 1]
        elif self.costModel == "broadcast":
            print("broadcast cost model")
            constraint = [(I - 2*EL + EL2 - J - s*I)<<0, (I - 2*EL + EL2 - J + s*I)>>0, En <= budget*num_nodes, sum_p == 1]
        else:
            print("p2p cost model")
            constraint = [(I - 2*EL + EL2 - J - s*I)<<0, (I - 2*EL + EL2 - J + s*I)>>0, Ee <= budget*num_links, sum_p == 1]

        problem = cp.Problem(cp.Minimize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)
        print("rho -- s.value:{}".format(s.value))
        print("rho -- p.value:{}".format(p.value))

        # normalize
        res_p = np.array(p.value)
        res_p /= res_p.sum()
        
        #print out the results
        if self.approach =="topologySamplingPrediction":
            if self.costModel == "time":
                expectedMatching = 0
                for i in range(len(res_p)):
                    expectedMatching+= res_p[i]*self.num_matchings[i]
                print("budget:{}, base_graph_matching:{}, expectedMatching:{}".format(budget, num_matchings, expectedMatching))
                sorted_index = np.argsort(res_p)
                if len(res_p)>=20:
                    for i in range(len(res_p) - 1 , len(res_p) - 21, -1):
                        print("index:{}, rho:{}, prob:{}, num_matchings:{}".format(sorted_index[i], self.rhos_subG[sorted_index[i]], res_p[sorted_index[i]], self.num_matchings[sorted_index[i]]))
            elif self.costModel == "bandwidth":
                expectedBandwidth = 0
                for i in range(len(res_p)):
                    expectedBandwidth+= res_p[i]*self.num_bandwidth[i]
                print("budget:{}, base_graph_bandwidth:{}, expectedBandwidth:{}".format(budget, num_bandwidth, expectedBandwidth))
                sorted_index = np.argsort(res_p)
                if len(res_p)>=20:
                    for i in range(len(res_p) - 1 , len(res_p) - 21, -1):
                        print("index:{}, rho:{}, prob:{}, bandwidth:{}".format(sorted_index[i], self.rhos_subG[sorted_index[i]], res_p[sorted_index[i]], self.num_bandwidth[sorted_index[i]]))
            elif self.costModel == "broadcast":
                expectedNodes = 0
                for i in range(len(res_p)):
                    expectedNodes+= res_p[i]*self.active_nodes[i]
                print("budget:{}, base_graph_nodes:{}, expectedNodes:{}".format(budget, self.size, expectedNodes))
                sorted_index = np.argsort(res_p)
                if len(res_p)>=20:
                    for i in range(len(res_p) - 1 , len(res_p) - 21, -1):
                        print("index:{}, rho:{}, prob:{}, active_nodes:{}".format(sorted_index[i], self.rhos_subG[sorted_index[i]], res_p[sorted_index[i]], self.active_nodes[sorted_index[i]]))
            else:
                expectedEdges = 0
                for i in range(len(res_p)):
                    expectedEdges+= res_p[i]*self.len_edges_subG[i]
                print("budget:{}, base_graph_edges:{}, expectedEdges:{}".format(budget, num_links, expectedEdges))
                sorted_index = np.argsort(res_p)
                if len(res_p)>=20:
                    for i in range(len(res_p) - 1, len(res_p) - 21, -1):
                        print("index:{}, rho:{}, prob:{}, active_edges:{}".format(sorted_index[i], self.rhos_subG[sorted_index[i]], res_p[sorted_index[i]], self.len_edges_subG[sorted_index[i]]))
        return res_p, s.value

    def getMixingMatrixs(self):
        resW = []
        num_nodes = self.size
        I = np.eye(num_nodes)
        for i in range(len(self.L_matrices)):
            resW.append(I - self.L_matrices[i])
        return resW

    def drawer(self):
        """
        input graph: list[list[tuples]]
                     [graph1, graph2,...]
                     graph: [edge1, edge2, ...]
                     edge: (node1, node2)
        output connect: matrix: [[]]
        """
        connect = []
        for graph in self.subGraphs:
            new_connect = [[] for i in range(self.size)]
            for edge in graph:
                node1, node2 = edge[0], edge[1]
                new_connect[node1].append(node2)
                new_connect[node2].append(node1)
            # print(new_connect)
            connect.append(new_connect)
        return connect


    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers
                     so that the activation flags are same
        """
        num_subgraphs = len(self.L_matrices)
        if self.seq:
            print("select random topologies, and repeat sequences")
            selected0 = list(np.random.choice(num_subgraphs, 1, p=self.probabilities))
            selected_ = list(np.random.choice(num_subgraphs, self.num_batches, p=self.probabilities))
            selected = selected0 + selected_*self.epoch
            if iterations != len(selected):
                print("iterations != len(selected)")
                raise
        else:
            selected = list(np.random.choice(num_subgraphs, iterations, p=self.probabilities))
        new_index = 0
        new_subG = list()
        new_subL = list()
        new_subRho = list()
        new_selected = list()
        mappings = {}

        for i in range(len(selected)):
            if selected[i] not in mappings:
                new_subG.append(self.subGraphs[selected[i]])
                new_subL.append(self.L_matrices[selected[i]])
                new_subRho.append(self.rhos_subG[selected[i]])
                mappings[selected[i]] = new_index
                new_index += 1
            new_selected.append(mappings[selected[i]])

        #update
        self.subGraphs = new_subG
        self.L_matrices = new_subL
        self.rhos_subG = new_subRho
        self.neighbors_info = self.drawer()
        self.activeTopology = list()
        for _ in range(iterations+1):
            self.activeTopology.append(np.zeros(self.L_matrices[0].shape))
        print("selected subG rho:{}".format(self.rhos_subG))
        new_num_subgraphs = len(self.L_matrices)
        print("# of selected subGrapg:{}".format(new_num_subgraphs))
        #raise
        rhos = list()
        flags = list()
        for i in range(len(new_selected)):
            f = np.zeros(new_num_subgraphs)
            f[new_selected[i]] = 1
            flags.append(f)
            rhos.append(self.rhos_subG[new_selected[i]])
            self.activeTopology[i] += self.L_matrices[new_selected[i]]
        print("first 30 rhos:{}".format(rhos[:30]))
        return flags, rhos


class TopologySamplingProcessor(GraphProcessor):
    def __init__(self, base_graph, commBudget, rank, size, iterations, sub_type, args, num_batches):
        super(TopologySamplingProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, sub_type, args)
        if sub_type != "resistGS":
            print("sub_type should be resistGS")
            raise
        self.ifBroadcast = args.broadcast
        self.seq = args.seq
        self.epoch = args.epoch
        self.num_batches = num_batches
        self.probabilities = self.getProbability()
        self.active_flags, self.rho  = self.set_flags(iterations + 1)
        self.mixingMatrixWs = self.getMixingMatrixs()


    def getProbability(self):
        num_subgraphs = len(self.L_matrices)
        num_links = len(list(self.base_graph.edges()))
        num_nodes = self.size

        # prepare matrices
        I = np.eye(num_nodes)
        J = np.ones((num_nodes, num_nodes))/num_nodes

        p = cp.Variable(num_subgraphs, nonneg=True)
        EL = p[0]*self.L_matrices[0]
        EL2 = p[0]*self.L_matrices[0].T@self.L_matrices[0]
        Ee = p[0]*self.len_edges_subG[0]
        En = p[0]*self.active_nodes[0]
        sum_p = p[0]
        for i in range(num_subgraphs-1):
            EL += p[i+1]*self.L_matrices[i+1]
            EL2 += p[i+1]*self.L_matrices[i+1].T@self.L_matrices[i+1]
            Ee += p[i+1]*self.len_edges_subG[i+1]
            En += p[i+1]*self.active_nodes[i+1]
            sum_p += p[i+1]

        s = cp.Variable()
        obj_fn = s
        if self.ifBroadcast:
            constraint = [(I - 2*EL + EL2 - J - s*I)<<0, (I - 2*EL + EL2 - J + s*I)>>0, En <= self.commBudget*num_nodes, sum_p == 1]
        else:
            constraint = [(I - 2*EL + EL2 - J - s*I)<<0, (I - 2*EL + EL2 - J + s*I)>>0, Ee <= self.commBudget*num_links, sum_p == 1]

        problem = cp.Problem(cp.Minimize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)
        print("rho -- s.value:{}".format(s.value))
        print("rho -- p.value:{}".format(p.value))
        print("rho -- p.value[0]:{}".format(p.value[0]))
        print("rho -- p.value[1]:{}".format(p.value[1]))
        # normalize

        res_p = np.array(p.value)
        res_p /= res_p.sum()

        return res_p

    def getMixingMatrixs(self):
        resW = []
        num_nodes = self.size
        I = np.eye(num_nodes)
        for i in range(len(self.L_matrices)):
            resW.append(I - self.L_matrices[i])
        return resW

    def drawer(self):
        """
        input graph: list[list[tuples]]
                     [graph1, graph2,...]
                     graph: [edge1, edge2, ...]
                     edge: (node1, node2)
        output connect: matrix: [[]]
        """
        connect = []
        for graph in self.subGraphs:
            new_connect = [[] for i in range(self.size)]
            for edge in graph:
                node1, node2 = edge[0], edge[1]
                new_connect[node1].append(node2)
                new_connect[node2].append(node1)
            # print(new_connect)
            connect.append(new_connect)
        return connect

    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers
                     so that the activation flags are same
        """
        num_subgraphs = len(self.L_matrices)
        if self.seq:
            print("select random topologies, and repeat sequences")
            selected0 = list(np.random.choice(num_subgraphs, 1, p=self.probabilities))
            selected_ = list(np.random.choice(num_subgraphs, self.num_batches, p=self.probabilities))
            selected = selected0 + selected_*self.epoch
            if iterations != len(selected):
                print("iterations != len(selected)")
                raise
        else:
            selected = list(np.random.choice(num_subgraphs, iterations, p=self.probabilities))
        new_index = 0
        new_subG = list()
        new_subL = list()
        new_subRho = list()
        new_selected = list()
        mappings = {}

        for i in range(len(selected)):
            if selected[i] not in mappings:
                new_subG.append(self.subGraphs[selected[i]])
                new_subL.append(self.L_matrices[selected[i]])
                new_subRho.append(self.rhos_subG[selected[i]])
                mappings[selected[i]] = new_index
                new_index += 1
            new_selected.append(mappings[selected[i]])

        #update
        self.subGraphs = new_subG
        self.L_matrices = new_subL
        self.rhos_subG = new_subRho
        self.neighbors_info = self.drawer()
        #print("iter=5 selected id:{}".format(new_selected[5]))
        #print("iter=5 subG:{}".format(self.subGraphs[new_selected[5]]))
        #print("iter=5 rho:{}".format(self.rhos_subG[new_selected[5]]))
        print("selected subG rho:{}".format(self.rhos_subG))
        new_num_subgraphs = len(self.L_matrices)
        print("# of selected subGrapg:{}".format(new_num_subgraphs))
        #raise
        rhos = list()
        flags = list()
        for i in range(len(new_selected)):
            f = np.zeros(new_num_subgraphs)
            f[new_selected[i]] = 1
            flags.append(f)
            rhos.append(self.rhos_subG[new_selected[i]])
        print("first 30 rhos:{}".format(rhos[:30]))
        return flags, rhos

class ResistanceSamplingProcessor(GraphProcessor):

    def __init__(self, base_graph, commBudget, rank, size, iterations, sub_type, args):
        super(ResistanceSamplingProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, sub_type, args)
        if sub_type != "links":
            print("sub_type should be links")
            raise
        if self.size != 33 or commBudget != 0.5:
            print("parameters in getMixingMatrix must be modified ")
            raise
        #parameters
        """
        self.l = 100000
        self.e = 0.275
        self.top = 5
        """
        """
        self.l = 50000
        self.e = 0.18
        self.top = 5
        """
        self.l = 100000
        self.e = args.epsilon
        self.top = args.top

        self.probabilities = np.ones(len(self.subGraphs))
        #print("self.subGraphs: {}".format(self.subGraphs))
        self.mixingMatrixWs, rhos = self.getMixingMatrixs()
        print("rhos: {}".format(rhos))
        #print("self.mixingMatrixWs: {}".format(self.mixingMatrixWs))
        self.rho = [rhos[i%self.top] for i in range(iterations + 1)]
        self.active_flags = self.set_flags(iterations + 1)
        self.neighbors_info = self.drawer()

    def getMixingMatrixs(self):
        resW = []
        resRho = []

        #get the heuristic solution
        #heuMatrixW, heuRho = self.getMixingMatrix()
        #resW.append(heuMatrixW)
        #resRho.append(heuRho)

        sparsifier.get_optimal_weight(self.base_graph)
        q = sparsifier.get_parameter_q(self.e,self.size)
        rho = []
        reweights = []
        for _ in range(self.l):
            LH, reweighted = sparsifier.Sparsify(self.base_graph,q)
            reweights.append(reweighted)
            eigvals_LH = LA.eigh(LH)[0]
            rho.append(max(1-eigvals_LH[1], eigvals_LH[-1] - 1)**2)

        B = sparsifier.get_signed_incidence_matrix(self.base_graph)
        #for _ in range(self.top - 1):
        for _ in range(self.top):
            i = np.argmin(rho)
            resW.append(np.identity(self.size) - B@np.diag(reweights[i])@B.transpose())
            resRho.append(rho[i])
            del(rho[i])
            del(reweights[i])
        return resW, resRho

    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers
                     so that the activation flags are same

        """
        flags = list()
        for i in range(len(self.L_matrices)):
            flags.append(np.random.binomial(1, self.probabilities[i], iterations))

        return [list(x) for x in zip(*flags)]

    def getMixingMatrix(self):
        rmE = int((1-self.commBudget)*len(self.subGraphs))
        if len(self.subGraphs) - rmE < self.size -1:
            print("budget is too small")
            raise

        def getrho(edges, numberNodes):
            #mixing matrix optimization
            numberLinks = len(edges)
            E = np.zeros((numberLinks, numberNodes))
            for i in range(numberLinks):
                E[i][edges[i][0][0]] = -1
                E[i][edges[i][0][1]] = 1

            I = np.eye(numberNodes)
            J = np.ones((numberNodes, numberNodes))/numberNodes
            # decision variables
            weights = cp.Variable(numberLinks, nonneg=True)
            L = E.T @ cp.diag(weights) @ E
            s = cp.Variable()
            constraints = [(I - L - J - s*I)<<0,(I - L - J + s*I)>>0]
            obj_fn = s
            problem = cp.Problem(cp.Minimize(obj_fn), constraints)
            #problem.solve()
            problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

            rho = problem.value*problem.value
            #print("rho:{}".format(rho))
            return rho, list(weights.value)

        edges = self.subGraphs
        for i in range(rmE):
            rho, w = getrho(edges, self.size)
            #print("rho:{}".format(rho))
            #print("W:{}".format(w))
            #print("np.argmin(w):{}".format(np.argmin(w)))
            del(edges[np.argmin(w)])
        rho, w = getrho(edges, self.size)
        #print("opt rho:{}".format(rho))
        #reconstruct grahp
        numberLinks = len(edges)
        numberNodes = self.size
        resProbabilities = np.ones(numberLinks)
        H = nx.Graph()
        for i in range(len(edges)):
            H.add_edges_from(edges[i])
        self.base_graph = H
        self.subGraphs = edges
        self.L_matrices = self.graphToLaplacian()
        self.neighbors_info = self.drawer()
        numberLinks = len(edges)
        E = np.zeros((numberLinks, numberNodes))
        for i in range(numberLinks):
            E[i][edges[i][0][0]] = -1
            E[i][edges[i][0][1]] = 1
        optL = E.T @ np.diag(w) @ E
        resW = np.identity(numberNodes) - optL
        #print("rho:{}".format(getrho(self.subGraphs, self.size)))
        #testr, testw = getrho(self.subGraphs, numberNodes)
        print("optimal w:{}".format(resW))
        print("rho:{}".format(rho))
        return resW, rho

class FixedExactOptimalProcessor(GraphProcessor):

    def __init__(self, base_graph, commBudget, rank, size, iterations, sub_type, args):
        super(FixedExactOptimalProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, sub_type)
        if sub_type != "links":
            print("sub_type should be links")
        if args.alg == "unweightedGS":
            self.weights, self.mixingMatrixW, rho,  self.probabilities = self.getMixingMatrixUnweighted()
        else:
            self.weights, self.mixingMatrixW, rho,  self.probabilities = self.getMixingMatrix()
        self.rho = [rho for _ in range(iterations + 1)]
        self.active_flags = self.set_flags(iterations + 1)
        self.neighbors_info = self.drawer()
    def getMixingMatrix(self):
        rmE = int((1-self.commBudget)*len(self.subGraphs))
        if len(self.subGraphs) - rmE < self.size -1:
            print("budget is too small")
            raise

        def getrho(edges, numberNodes):
            #mixing matrix optimization
            numberLinks = len(edges)
            E = np.zeros((numberLinks, numberNodes))
            for i in range(numberLinks):
                E[i][edges[i][0][0]] = -1
                E[i][edges[i][0][1]] = 1

            I = np.eye(numberNodes)
            J = np.ones((numberNodes, numberNodes))/numberNodes
            # decision variables
            weights = cp.Variable(numberLinks)
            L = E.T @ cp.diag(weights) @ E
            s = cp.Variable()
            constraints = [(I - L - J - s*I)<<0,(I - L - J + s*I)>>0]
            obj_fn = s
            problem = cp.Problem(cp.Minimize(obj_fn), constraints)
            #problem.solve()
            problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

            rho = problem.value*problem.value
            #print("rho:{}".format(rho))
            return rho, list(weights.value)

        edges = self.subGraphs
        for _ in range(rmE):
            rho, w = getrho(edges, self.size)
            #print("rho:{}".format(rho))
            #print("W:{}".format(w))
            #print("np.argmin(w):{}".format(np.argmin(w)))
            minweight = float("inf")

            for j in range(len(w)):
                if minweight > np.abs(w[j]):
                    minweight = np.abs(w[j])
                    argminweight = j
            del(edges[argminweight])
        rho, w = getrho(edges, self.size)
        #print("opt rho:{}".format(rho))
        #reconstruct grahp
        numberLinks = len(edges)
        numberNodes = self.size
        resProbabilities = np.ones(numberLinks)
        H = nx.Graph()
        for i in range(len(edges)):
            H.add_edges_from(edges[i])
        self.base_graph = H
        self.subGraphs = edges
        self.L_matrices = self.graphToLaplacian()
        self.neighbors_info = self.drawer()
        numberLinks = len(edges)
        E = np.zeros((numberLinks, numberNodes))
        for i in range(numberLinks):
            E[i][edges[i][0][0]] = -1
            E[i][edges[i][0][1]] = 1
        optL = E.T @ np.diag(w) @ E
        resW = np.identity(numberNodes) - optL
        #print("rho:{}".format(getrho(self.subGraphs, self.size)))
        #testr, testw = getrho(self.subGraphs, numberNodes)
        print("optimal w:{}".format(resW))
        print("rho:{}".format(rho))
        return w, resW, rho, resProbabilities

    def getMixingMatrixUnweighted(self):
        numberLinks = len(self.subGraphs)
        E = np.zeros((len(self.subGraphs), self.size))
        for i in range(len(self.subGraphs)):
            E[i][self.subGraphs[i][0][0]] = -1
            E[i][self.subGraphs[i][0][1]] = 1
        baseGraphL = self.L_matrices[0]
        for i in range(1, len(self.L_matrices)):
            baseGraphL += self.L_matrices[i]
        if not np.array_equal(baseGraphL, E.T @ E):
            print("ETE != L")
            print("ETE:{}".format(E.T @ E))
            print("L:{}".format(baseGraphL))
            raise

        u,s,vt = np.linalg.svd(E, full_matrices=True)
        rankE = np.linalg.matrix_rank(E)
        x = np.diag(s[:rankE]) @ u[:,:rankE].T
        r = int(self.commBudget*numberLinks)

        rankx = np.linalg.matrix_rank(x)
        if rankx!=x.shape[0]:
            print("x.shape:{}".format(x.shape[0]))
            print("rank:{}",format(rankx))
            raise
        l = x.shape[1]

        #get u, v matrix
        u,s,vt = np.linalg.svd(x, full_matrices=True)
        v = vt.T
        v = v[:,:rankx]
        v = v.T
        u = np.identity(l)

        #init
        s = np.zeros((x.shape[1],1))
        A = np.zeros((rankx, rankx))
        B = np.zeros((l,l))
        deltaL = 1
        deltaU = (1+np.sqrt(l/r))/(1-np.sqrt(rankx/r))

        def getPhiBottom(Lt, A):
            if not np.array_equal(A, A.T):
                print("A is not sym")
                print(A)
                raise
            lambdaA, _ = np.linalg.eigh(A)
            return sum([1/(lambdaA[i] - Lt) for i in range(rankx)])
        def getPhiTop(Ut, B):
            if not np.array_equal(B, B.T):
                print("B is not sym")
                print(B)
                raise
            lambdaB, _ = np.linalg.eigh(B)
            return sum([1/(Ut - lambdaB[i]) for i in range(l)])
        for t in range(r):
            print("t:{}".format(t))
            Lt = t - np.sqrt(r*rankx)
            Ut = deltaU*(t+np.sqrt(l*r))
            #verion 1
            #j = t % x.shape[1]
            #vj = v[:,j].reshape((v.shape[0], 1))
            #uj = u[:,j].reshape((u.shape[0], 1))
            #L = (vj.T @ np.linalg.matrix_power((A -(Lt+deltaL)*np.identity(rankx)), -2) @ vj )/(getPhiBottom(Lt+deltaL, A) - getPhiBottom(Lt, A)) - (vj.T @ np.linalg.matrix_power((A -(Lt+deltaL)*np.identity(rankx)), -1) @ vj )
            #U = (uj.T @ np.linalg.matrix_power(((Ut + deltaU)*np.identity(l) - B),-2) @ uj )/(getPhiTop(Ut, B) - getPhiTop(Ut+deltaU, B)) + (uj.T @ np.linalg.matrix_power(((Ut + deltaU)*np.identity(l) - B),-1) @ uj )
            #if U<=L:
            #    t_ = 2/(float(U)+float(L))
            #    s[j] = s[j] + t_
            #    A = A + t_*(vj @ vj.T)
            #    B = B + t_*(uj @ uj.T)
            #version 2
            found = False
            for j in range(x.shape[1]):
                vj = v[:,j].reshape((v.shape[0], 1))
                uj = u[:,j].reshape((u.shape[0], 1))
                L = (vj.T @ np.linalg.matrix_power((A -(Lt+deltaL)*np.identity(rankx)), -2) @ vj )/(getPhiBottom(Lt+deltaL, A) - getPhiBottom(Lt, A)) - (vj.T @ np.linalg.matrix_power((A -(Lt+deltaL)*np.identity(rankx)), -1) @ vj )
                U = (uj.T @ np.linalg.matrix_power(((Ut + deltaU)*np.identity(l) - B),-2) @ uj )/(getPhiTop(Ut, B) - getPhiTop(Ut+deltaU, B)) + (uj.T @ np.linalg.matrix_power(((Ut + deltaU)*np.identity(l) - B),-1) @ uj )
                if U<=L:
                    t_ = 2/(float(U)+float(L))
                    found = True
                    break
            #update
            if not found:
                print("U<=L can't be satisfied")
                raise

            s[j] = s[j] + t_
            A = A + t_*(vj @ vj.T)
            B = B + t_*(uj @ uj.T)
        print("s:{}".format(s))
        H = nx.Graph()
        Hsub = []
        for i in range(numberLinks):
            if s[i]!=0:
                H.add_edges_from(self.subGraphs[i])
                Hsub.append(self.subGraphs[i])

        self.base_graph = H
        self.subGraphs = Hsub
        self.L_matrices = self.graphToLaplacian()
        self.neighbors_info = self.drawer()

        #print("self.L_matrices: {}".format(self.L_matrices))
        #print("self.neighbors_info:{}".format(self.neighbors_info))

        #mixing matrix optimization
        numberLinks = len(self.subGraphs)
        numberNodes = self.size
        E = np.zeros((numberLinks, numberNodes))
        for i in range(numberLinks):
            E[i][self.subGraphs[i][0][0]] = -1
            E[i][self.subGraphs[i][0][1]] = 1

        I = np.eye(numberNodes)
        J = np.ones((numberNodes, numberNodes))/numberNodes
        # decision variables
        weights = cp.Variable(numberLinks, nonneg=True)
        L = E.T @ cp.diag(weights) @ E
        s = cp.Variable()
        constraints = [(I - L - J - s*I)<<0,(I - L - J + s*I)>>0]
        obj_fn = s
        problem = cp.Problem(cp.Minimize(obj_fn), constraints)
        #problem.solve()
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

        rho = problem.value*problem.value
        print("rho:{}".format(rho))
        #raise
        optL = E.T @ np.diag(list(weights.value)) @ E
        resW = np.identity(numberNodes) - optL
        return list(weights.value), resW, rho, np.ones(numberLinks)

    def getMixingMatrix_(self):
        numberLinks = len(self.subGraphs)
        numberNodes = self.size

        E = np.zeros((numberLinks, numberNodes))
        for i in range(numberLinks):
            E[i][self.subGraphs[i][0][0]] = -1
            E[i][self.subGraphs[i][0][1]] = 1
        # Laplacian matrix = E.T @ np.diag(weights) @ E

        baseGraphL = self.L_matrices[0]
        for i in range(1, len(self.L_matrices)):
            baseGraphL += self.L_matrices[i]
        if not np.array_equal(baseGraphL, E.T @ E):
            print("ETE != L")
            print("ETE:{}".format(E.T @ E))
            print("L:{}".format(baseGraphL))
            raise

        I = np.eye(numberNodes)
        J = np.ones((numberNodes, numberNodes))/numberNodes
        # decision variables
        weights = cp.Variable(numberLinks, nonneg=True)
        #alpha = cp.Variable(nonneg=True)
        #weights = np.ones(numberLinks)*alpha
        L = E.T @ cp.diag(weights) @ E
        s = cp.Variable()
        constraints = [(I - L - J - s*I)<<0,(I - L - J + s*I)>>0]
        obj_fn = s
        problem = cp.Problem(cp.Minimize(obj_fn), constraints)
        problem.solve()
        #optWeights = np.maximum(np.diag(weights.value),0.0000001)
        optWeights = np.diag(weights.value)
        optL = E.T @ optWeights @ E
        resWeights = optWeights
        #print("resWeights:{}".format(resWeights))
        #raise
        resW = I - optL
        rho = problem.value*problem.value
        resProbabilities = np.ones(numberLinks)
        print("rho:{}".format(rho))
        #raise

        if self.commBudget <1:
            d = self.commBudget*numberLinks/numberNodes
            if d <= 1:
                print("self.commBudget is too small")
                raise
            r = int(d*numberNodes)
            eigenval, eigenvec = np.linalg.eigh(optL)
            if eigenval[0] >1e-10:
                print("smallest eigenval:{}".format(eigenval[0]))
                print("the smallest eigenval of L is not 0")
                raise
            print("eigenval:{}".format(eigenval))
            print("eigenvec @ eigenvec.T:{}".format(eigenvec @ eigenvec.T))
            optLInversesqrt = np.sqrt(1/eigenval[1])* eigenvec[:,1].reshape(eigenvec.shape[0], 1) @ eigenvec[:,1].reshape(eigenvec.shape[0], 1).T
            #optLInversesqrt = (1/eigenval[1])* eigenvec[:,1].reshape(eigenvec.shape[0], 1) @ eigenvec[:,1].reshape(eigenvec.shape[0], 1).T
            print("numberLinks:{}".format(numberLinks))
            print("numberLinks:{}".format(numberLinks))
            print("len(eigenval):{}".format(len(eigenval)))
            for i in range(2, numberNodes):
                print("i:".format(i))
                optLInversesqrt += np.sqrt(1/eigenval[i])* eigenvec[:,i].reshape(eigenvec.shape[0], 1) @ eigenvec[:,i].reshape(eigenvec.shape[0], 1).T
                #optLInversesqrt += (1/eigenval[i])* eigenvec[:,i].reshape(eigenvec.shape[0], 1) @ eigenvec[:,i].reshape(eigenvec.shape[0], 1).T

            print("optL:{}".format(optL))
            print("optLInversesqrt  @ optL:{}".format( optLInversesqrt @ optL))
            if optLInversesqrt.shape != optL.shape:
                print("optLInversesqrt.shape:{}".format(optLInversesqrt.shape))
                print("optL.shape:{}".format(optL.shape))
                raise
            optWeights[4][4] = 0.4
            print("optWeights:{}".format(optWeights))

            V = optLInversesqrt @ E.T @ (optWeights**(1/2))
            print("V:{}".format(V))
            print("V @ V.T:{}".format(V @ V.T))
            print("eigenvec[:,i].reshape(eigenvec.shape[0], 1):{}".format(eigenvec[:,1].reshape(eigenvec.shape[0], 1)))
            print("V @ V.T @ eigenvec:{}".format( V @ V.T @ eigenvec[:,1].reshape(eigenvec.shape[0], 1)))

            #print("optLInversesqrt @ E.T :{}".format(optLInversesqrt @ E.T ))
            #print("optWeights:{}".format(optWeights))
            #print("optWeights**(1/2):{}".format(optWeights**(1/2)))
            #print("V:{}".format(V))
            #print("V.shape:{}".format(V.shape))
            #raise
            kk = (d + 2*np.sqrt(d) + 1)/(d - 2*np.sqrt(d) + 1)
            def getPhiBottom(l, M):
                if not np.array_equal(M, M.T):
                    print("M is not sym")
                    print(M)
                    raise
                lambda_, _ = np.linalg.eigh(M)
                return sum([1/(lambda_[i] - l) for i in range(numberNodes)])
            def getPhiTop(u, M):
                if not np.array_equal(M, M.T):
                    print("M is not sym")
                    print(M)
                    raise
                lambda_, _ = np.linalg.eigh(M)
                return sum([1/(u - lambda_[i]) for i in range(numberNodes)])
            s = np.zeros((numberLinks))
            A = np.zeros((numberNodes, numberNodes))
            deltaL = 0.6
            deltaU = (np.sqrt(d)+1)/(np.sqrt(d)-1)
            epL = 1/np.sqrt(d)
            epU = (np.sqrt(d) - 1)/(d+np.sqrt(d))
            Lt = -numberNodes/epL-0.1
            Ut = numberNodes/epU +0.1
            for t in range(r):
                print("t:{}".format(t))
                #check if vj exists
                eigenval, eigenvec = np.linalg.eigh(A)
                if eigenval[-1] >= Ut:
                    print("constraint 1 failed")
                    print("eigenval[-1]:{}".format(eigenval[-1]))
                    print("Ut:{}".format(Ut))
                    raise
                if eigenval[0] <= Lt:
                    print("constraint 2 failed")
                    print("eigenval[0]:{}".format(eigenval[0]))
                    print("Lt:{}".format(Lt))
                    raise
                if getPhiTop(Ut, A) > epU:
                    print("constraint 3 failed")
                    print("getPhiTop(Ut, A):{}".format(getPhiTop(Ut, A)))
                    print("epU:{}".format(epU))
                    raise
                if getPhiBottom(Lt, A) > epL:
                    print("constraint 4 failed")
                    print("getPhiBottom(Lt, A):{}".format(getPhiBottom(Lt, A)))
                    print("epL:{}".format(epL))
                    raise
                if (1/deltaU + epU)< 0:
                    print("constraint 5 failed")
                    print("1/deltaU + epU:{}".format(1/deltaU + epU))
                    raise
                if (1/deltaU + epU) > (1/deltaL - epL):
                    print("constraint 5 failed")
                    print("(1/deltaU + epU):{}".format(1/deltaU + epU))
                    print("(1/deltaL - epL):{}".format((1/deltaL - epL)))
                    raise

                found = False

                for j in range(numberLinks):
                    vj = V[:,j].reshape((V.shape[0], 1))
                    #print("vj.shape:{}".format(vj.shape))
                    #print("vj:{}".format(vj))
                    #print("L0:{}".format( np.linalg.matrix_power((A -(Lt+deltaL)*np.identity(numberNodes)), -2)@ vj))
                    #print("L1:{}".format((vj.T @ np.linalg.matrix_power((A -(Lt+deltaL)*np.identity(numberNodes)), -2) @ vj )))
                    #print("L2:{}".format((getPhiBottom(Lt+deltaL, A) - getPhiBottom(Lt, A)) ))
                    #print("L3:{}".format( (vj.T @ np.linalg.matrix_power((A -(Lt+deltaL)*np.identity(numberNodes)), -1) @ vj )))
                    L = (vj.T @ np.linalg.matrix_power((A -(Lt+deltaL)*np.identity(numberNodes)), -2) @ vj )/(getPhiBottom(Lt+deltaL, A) - getPhiBottom(Lt, A)) - (vj.T @ np.linalg.matrix_power((A -(Lt+deltaL)*np.identity(numberNodes)), -1) @ vj )
                    U = (vj.T @ np.linalg.matrix_power(((Ut + deltaU)*np.identity(numberNodes) - A),-2) @ vj )/(getPhiTop(Ut, A) - getPhiTop(Ut+deltaU, A)) + (vj.T @ np.linalg.matrix_power(((Ut + deltaU)*np.identity(numberNodes) - A),-1) @ vj )
                    print("j:{}".format(j))
                    print("L:{}".format(L))
                    print("U:{}".format(U))

                    TLA = np.linalg.matrix_power((A -(Lt+deltaL)*np.identity(numberNodes)), -2)
                    TUA = np.linalg.matrix_power(((Ut + deltaU)*np.identity(numberNodes) - A),-2)
                    #print("TLA:{}".format(TLA))
                    #print("TLA dot V @ V.T :{}".format(np.sum(TLA * (V @ V.T))))
                    #print("np.trace(TLA):{}".format(np.trace(TLA)))
                    #print("TUA:{}".format(TUA))
                    #print("TUA dot V @ V.T :{}".format(np.sum(TUA * (V @ V.T))))
                    #print("np.trace(TUA):{}".format(np.trace(TUA)))
                    #print("TUA @ V @ V.T :{}".format(TUA @ V @ V.T))
                    #raise
                    #if t ==0:
                    #    print("A:{}".format(A))
                    #    print("getPhiBottom(Lt, A):{}".format(getPhiBottom(Lt, A)))
                    #    print("el:{}".format(-numberNodes/Lt))
                    #    print("(getPhiTop(Ut, A):{}".format(getPhiTop(Ut, A)))
                    #    print("eu:{}".format(numberNodes/Ut))
                    #    raise
                    if U<=L and (not (L==0 and U==0)):
                        print("L:{}".format(L))
                        print("U:{}".format(U))
                        #ratio = 0.5
                        t_ = 2/(float(U)+float(L))
                        #t_ = ratio*(1/float(U))+(1-ratio)*(1/float(L))
                        if 1/t_ >L or 1/t_ < U:
                            print("t_ is not qualified")
                            raise
                        print("t_:{}".format(t_))
                        found = True
                        break
                if not found:
                    print("U<=L can't be satisfied")
                    raise
                #update
                if t_ <0:
                    print("t_:{}".format(t_))
                    raise
                s[j] = s[j] + t_
                A = A + t_*(vj @ vj.T)
                eigenval, eigenvec = np.linalg.eig(A)
                print("eigenval:{}".format(eigenval))
                print("lambdaMax/lambdaMin:{}".format(max(eigenval)/min(eigenval)))
                print("K:{}".format(kk))
                Lt += deltaL
                Ut += deltaU

            LH = E.T @ optWeights**(1/2) @ np.diag(s) @ optWeights**(1/2) @ E
            eigenval, eigenvec = np.linalg.eigh(LH)
            if eigenval[0] >1e-10:
                print("the smallest eigenval of LH is not 0")
                raise
            alpha = 2/(eigenval[1] + eigenval[-1])
            print("alpha:{}".format(alpha))
            optLH = alpha*LH
            resW = np.identity(numberNodes) - optLH

            #resWeights = alpha*optWeights @ np.diag(s)
            #resWeights =
            rho = max(1-alpha*eigenval[1], alpha*eigenval[-1]-1)**2
            for i in range(numberLinks):
                if s[i]==0:
                    resProbabilities = 0
            print("resWeights:{}".format(resWeights))
            print("resW:{}".format(resW))
            print("rho:{}".format(rho))
            print("resProbabilities:{}".format(resProbabilities))
            raise

        return resWeights, resW, rho, resProbabilities

    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers
                     so that the activation flags are same

        """
        flags = list()
        for i in range(len(self.L_matrices)):
            flags.append(np.random.binomial(1, self.probabilities[i], iterations))


        return [list(x) for x in zip(*flags)]

class FixedOptimalProcessor(GraphProcessor):

    def __init__(self, base_graph, commBudget, rank, size, iterations, sub_type, ca, cb):
        super(FixedOptimalProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, sub_type)
        if sub_type != "links":
            print("sub_type should be links")
            raise
        self.ca = ca
        self.cb = cb
        self.probabilities = self.getProbability()
        print("probabilities:{}".format(self.probabilities))
        self.neighbor_weight, rho = self.getAlpha()
        self.rho = [rho for _ in range(iterations + 1)]
        self.active_flags = self.set_flags(iterations + 1)
        self.neighbors_info = self.drawer()
    def getProbability(self):
        E = np.zeros((len(self.subGraphs), self.size))
        for i in range(len(self.subGraphs)):
            E[i][self.subGraphs[i][0][0]] = -1
            E[i][self.subGraphs[i][0][1]] = 1
        baseGraphL = self.L_matrices[0]
        for i in range(1, len(self.L_matrices)):
            baseGraphL += self.L_matrices[i]
        if not np.array_equal(baseGraphL, E.T @ E):
            print("ETE != L")
            print("ETE:{}".format(E.T @ E))
            print("L:{}".format(baseGraphL))
            raise

        u,s,vt = np.linalg.svd(E, full_matrices=True)
        rankE = np.linalg.matrix_rank(E)
        x = np.diag(s[:rankE]) @ u[:,:rankE].T
        r = int((self.commBudget - self.size*self.ca)/(2*self.cb))

        rankx = np.linalg.matrix_rank(x)
        if rankx!=x.shape[0]:
            print("x.shape:{}".format(x.shape[0]))
            print("rank:{}",format(rankx))
            raise
        l = x.shape[1]

        #get u, v matrix
        u,s,vt = np.linalg.svd(x, full_matrices=True)
        v = vt.T
        v = v[:,:rankx]
        v = v.T
        u = np.identity(l)

        #init
        s = np.zeros((x.shape[1],1))
        A = np.zeros((rankx, rankx))
        B = np.zeros((l,l))
        deltaL = 1
        deltaU = (1+np.sqrt(l/r))/(1-np.sqrt(rankx/r))

        def getPhiBottom(Lt, A):
            if not np.array_equal(A, A.T):
                print("A is not sym")
                print(A)
                raise
            lambdaA, _ = np.linalg.eigh(A)
            return sum([1/(lambdaA[i] - Lt) for i in range(rankx)])
        def getPhiTop(Ut, B):
            if not np.array_equal(B, B.T):
                print("B is not sym")
                print(B)
                raise
            lambdaB, _ = np.linalg.eigh(B)
            return sum([1/(Ut - lambdaB[i]) for i in range(l)])
        for t in range(r):
            Lt = t - np.sqrt(r*rankx)
            Ut = deltaU*(t+np.sqrt(l*r))
            found = False
            for j in range(x.shape[1]):
                vj = v[:,j].reshape((v.shape[0], 1))
                uj = u[:,j].reshape((u.shape[0], 1))
                L = (vj.T @ np.linalg.matrix_power((A -(Lt+deltaL)*np.identity(rankx)), -2) @ vj )/(getPhiBottom(Lt+deltaL, A) - getPhiBottom(Lt, A)) - (vj.T @ np.linalg.matrix_power((A -(Lt+deltaL)*np.identity(rankx)), -1) @ vj )
                U = (uj.T @ np.linalg.matrix_power(((Ut + deltaU)*np.identity(l) - B),-2) @ uj )/(getPhiTop(Ut, B) - getPhiTop(Ut+deltaU, B)) + (uj.T @ np.linalg.matrix_power(((Ut + deltaU)*np.identity(l) - B),-1) @ uj )
                if U<=L:
                    t_ = 2/(float(U)+float(L))
                    found = True
                    break
            #update
            if not found:
                print("U<=L can't be satisfied")
                raise
            s[j] = s[j] + t_
            A = A + t_*(vj @ vj.T)
            B = B + t_*(uj @ uj.T)
        pro = np.zeros(len(self.subGraphs))
        for i in range(len(s)):
            if s[i]!=0:
                pro[i] = 1

        if sum(pro) > r:
            print("sum(pro) > r")
            print("r:{}".format(r))
            print("sum(pro):{}".format(sum(pro)))
            raise
        #print(self.subGraphs)
        #print(pro)
        #raise
        return pro

    def getAlpha(self):
        num_subgraphs = len(self.L_matrices)
        num_nodes = self.size

        # prepare matrices
        I = np.eye(num_nodes)
        J = np.ones((num_nodes, num_nodes))/num_nodes

        mean_L = np.zeros((num_nodes,num_nodes))
        var_L = np.zeros((num_nodes,num_nodes))
        for i in range(num_subgraphs):
            val = self.probabilities[i]
            mean_L += self.L_matrices[i]*val
            var_L += self.L_matrices[i]*(1-val)*val

        # SDP for mixing weight
        a = cp.Variable()
        b = cp.Variable()
        s = cp.Variable()
        obj_fn = s
        constraint = [(1-s)*I - 2*a*mean_L-J + b*(np.dot(mean_L,mean_L)+2*var_L) << 0, a>=0, s>=0, b>=0, cp.square(a) <= b]
        problem = cp.Problem(cp.Minimize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)
        return  float(a.value), float(s.value)


    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers
                     so that the activation flags are same

        """
        flags = list()
        for i in range(len(self.L_matrices)):
            flags.append(np.random.binomial(1, self.probabilities[i], iterations))



class FixedEachOptimalProcessor(GraphProcessor):

    def __init__(self, base_graph, commBudget, rank, size, iterations, sub_type, args):
        super(FixedEachOptimalProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, sub_type)
        if sub_type != "links":
            print("sub_type should be links")
        self.probabilities = self.getProbability()
        self.active_flags = self.set_flags(iterations + 1)
        self.mixingMatrixWs, self.rho = self.getMixingMatrix(iterations+1)

    def getMixingMatrix(self, iterations):

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
            L = E.T @ cp.diag(weights) @ E
            s = cp.Variable()
            constraints = [(I - L - J - s*I)<<0,(I - L - J + s*I)>>0]
            obj_fn = s
            problem = cp.Problem(cp.Minimize(obj_fn), constraints)
            #problem.solve()
            problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

            rho = problem.value*problem.value
            #print("rho:{}".format(rho))
            return rho, list(weights.value)

        numberNodes = self.size
        numberLinks = len(self.subGraphs)
        resMixingMWs = []
        resWeights = []
        resEigenvals = []
        rhos = []
        for itr in range(iterations):
            #mapping = { edge index of subgraph: edge index of base graph}
            mapping = {}
            subgraphEdges = []
            sub_link_id = 0
            #subgraph id is link id, when subtype=="links"
            for base_link_id, flag in enumerate(self.active_flags[itr]):
                if flag != 0:
                    subgraphEdges.append((self.subGraphs[base_link_id][0]))
                    mapping[sub_link_id] = base_link_id
                    sub_link_id += 1
            rho, subW = getrho(subgraphEdges, numberNodes)
            weights = np.zeros(numberLinks)
            for sub_link_id, w in enumerate(subW):
                weights[mapping[sub_link_id]] = w

            E = np.zeros((numberLinks, numberNodes))
            for i in range(numberLinks):
                E[i][self.subGraphs[i][0][0]] = -1
                E[i][self.subGraphs[i][0][1]] = 1
            optL = E.T @ np.diag(weights) @ E
            eigenval, eigenvec = np.linalg.eigh(optL)
            resEigenvals.append((eigenval[1], eigenval[-1]))
            mixingMW = np.identity(numberNodes) - optL
            resWeights.append(weights)
            resMixingMWs.append(mixingMW)
            rhos.append(rho)
        #print("resEigenvals:{}".format(resEigenvals))
        #print("self.subGrahps:{}".format(self.subGraphs))
        #print("resWeights:{}".format(resWeights))
        print("rhos:{}".format(rhos))
        #raise
        return resMixingMWs, rhos

    def getProbability(self):
        num_subgraphs = len(self.L_matrices)
        p = cp.Variable(num_subgraphs)
        L = p[0]*self.L_matrices[0]
        for i in range(num_subgraphs-1):
            L += p[i+1]*self.L_matrices[i+1]
        eig = cp.lambda_sum_smallest(L, 2)
        sum_p = p[0]
        for i in range(num_subgraphs-1):
            sum_p += p[i+1]

        # cvx optimization for activation probabilities
        obj_fn = eig
        constraint = [sum_p <= num_subgraphs*self.commBudget, p>=0, p<=1]
        problem = cp.Problem(cp.Maximize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

        # get solution
        tmp_p = p.value
        originActivationRatio = np.zeros((num_subgraphs))
        for i, pval in enumerate(tmp_p):
            originActivationRatio[i] = np.real(float(pval))

        return np.maximum(np.minimum(originActivationRatio, 1),0)

    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers
                     so that the activation flags are same

        """
        flags = list()
        for i in range(len(self.L_matrices)):
            flags.append(np.random.binomial(1, self.probabilities[i], iterations))


        return [list(x) for x in zip(*flags)]
