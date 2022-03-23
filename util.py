import os
import numpy as np
import time
import argparse

from mpi4py import MPI
from math import ceil
from random import Random
import networkx as nx
from pygraphml import Graph
from pygraphml import GraphMLParser

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
import simpleModel
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models
from comm_helpers import flatten_tensors, unflatten_tensors
from operator import itemgetter, attrgetter, methodcaller
from models import *
from collections import defaultdict
#import GraphPreprocess
import resnet
import wrn
from scipy.io import loadmat
from pygraphml import Graph

def loadgraph(addr):
    gphml = Graph()
    gmatrix = loadmat(addr)["A"].toarray()
    for i in range(len(gmatrix)):
        n = gphml.add_node()
        n.id = str(i)
    for i in range(len(gmatrix)):
        for j in range(i+1, len(gmatrix)):
            if gmatrix[i,j] == 1:
                gphml.add_edge_by_id(str(i), str(j))
    return gphml

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, args, data, sizes=[0.7, 0.2, 0.1], isNonIID=False, seed=1234):
        #print("sizes:{}".format(sizes))
        #print("seed:{}".format(seed))
        print("isNonIID:{}".format(isNonIID))
        if isNonIID:
            print("non iid level:{}".format(args.noniidlevel))
        #raise
        self.args = args
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        print("Total data len:{}".format(data_len))
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)


        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

        if isNonIID:
            self.partitions = self.__getNonIIDdata__(data, sizes, seed)

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


    def __getNonIIDdata__(self, data, sizes, seed):
        labelList = data.targets
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = defaultdict(list)
        for label, idx in a:
            labelIdxDict[label].append(idx)
        #print("labelIdxDict:{}".format(labelIdxDict))
        print("# labels:{}".format(len(labelIdxDict)))
        if self.args.noniidlevel is None:
            orderedDataIndex = []
            for label, idxs in sorted(labelIdxDict.items()):
                orderedDataIndex += idxs    
            data_len = len(data)
            if len(orderedDataIndex) !=data_len:
                print("orderedDataIndex is wrong")
                raise
            partitions= []
            for frac in sizes:
                part_len = int(frac* data_len)
                partitions.append(orderedDataIndex[0:part_len])
                orderedDataIndex = orderedDataIndex[part_len:]

        else:
            labelNum = len(labelIdxDict)
            labelNameList = [key for key in labelIdxDict]
            labelIdxPointer = [0] * labelNum
            partitions = [list() for i  in range(len(sizes))]
            eachPartitionLen= int(len(labelList)/len(sizes))
            majorLabelNumPerPartition = ceil(labelNum/len(partitions))
            basicLabelRatio = self.args.noniidlevel
    
            if (len(partitions)/labelNum)*self.args.noniidlevel > 1:
                print("basicLabelRatio is too large")
                print("Some data with same label are dispensed to {} workers, and each worker has {} percent".format(len(partitions)/labelNum, basicLabelRatio))
                raise
            print("labelNum:{}".format(labelNum))
            interval = 1
            labelPointer = 0
    
            #basic part
            for partPointer in range(len(partitions)):
                requiredLabelList = list()
                for _ in range(majorLabelNumPerPartition):
                    requiredLabelList.append(labelPointer)
                    labelPointer += interval
                    if labelPointer > labelNum - 1:
                        labelPointer = 0
                        #labelPointer = interval
                        #interval += 1
                print("requiredLabelList:{}".format(requiredLabelList))
                for labelIdx in requiredLabelList:
                    start = labelIdxPointer[labelIdx]
                    idxIncrement = int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
                    partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                    labelIdxPointer[labelIdx] += idxIncrement
    
            #random part
            remainLabels = list()
            for labelIdx in range(labelNum):
                remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
            rng.shuffle(remainLabels)
            for partPointer in range(len(partitions)):
                idxIncrement = eachPartitionLen - len(partitions[partPointer])
                if idxIncrement <0:
                    print("basicLabelRatio is too large")
                    raise
                partitions[partPointer].extend(remainLabels[:idxIncrement])
                rng.shuffle(partitions[partPointer])
                remainLabels = remainLabels[idxIncrement:]
        return partitions
        

def partition_dataset(rank, size, args):
    print('==> load train data')
    if args.dataset =="fashion-mnist":
        transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        trainset = torchvision.datasets.FashionMNIST(
            root=args.datasetRoot,
            train=True,
            download=True,
            transform=transform_train
        )
        if args.NotEvenPart:
            partition_sizes = np.random.randint(1,6, size=size)
            partition_sizes = args.datafrac*(partition_sizes/sum(partition_sizes))
            for _ in range(args.shuffleTimes):
                np.random.shuffle(partition_sizes)
        else:
            partition_sizes = [args.datafrac*(1.0 / size) for _ in range(size)]
        print("partition_sizes:{}".format(partition_sizes))
        partition = DataPartitioner(args, trainset, partition_sizes, isNonIID = args.noniid)
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition,
                                                batch_size=args.bs,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=0)
        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        testset = torchvision.datasets.FashionMNIST(
            root=args.datasetRoot,
            train=False,
            download=True,
            transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=args.bs,
                                                shuffle=False,
                                                num_workers=0)

    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=args.datasetRoot,
                                                train=True,
                                                download=True,
                                                transform=transform_train)

        if args.NotEvenPart:
            partition_sizes = np.random.randint(1,6, size=size)
            partition_sizes = args.datafrac*(partition_sizes/sum(partition_sizes))
            for _ in range(args.shuffleTimes):
                np.random.shuffle(partition_sizes)
        else:
            partition_sizes = [args.datafrac*(1.0 / size) for _ in range(size)]
        print("partition_sizes:{}".format(partition_sizes))
        partition = DataPartitioner(args, trainset, partition_sizes, isNonIID = args.noniid)
        #print("rank:{}, local data:{}".format(rank, partition.partitions[rank]))
        
        #np.savetxt(args.savePath + args.name + '_' + args.model+'/rank' + str(rank) + '-localdata.log', partition.partitions[rank], delimiter=',')
        
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition,
                                                batch_size=args.bs,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=0)

        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root=args.datasetRoot,
                                            train=False,
                                            download=True,
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=args.bs,
                                                shuffle=False,
                                                num_workers=0)

    if args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=args.datasetRoot,
                                                train=True,
                                                download=True,
                                                transform=transform_train)

        if args.NotEvenPart:
            partition_sizes = np.random.randint(1,6, size=size)
            partition_sizes = args.datafrac*(partition_sizes/sum(partition_sizes))
            for _ in range(args.shuffleTimes):
                np.random.shuffle(partition_sizes)
        else:
            partition_sizes = [args.datafrac*(1.0 / size) for _ in range(size)]
        print("partition_sizes:{}".format(partition_sizes))
        partition = DataPartitioner(args, trainset, partition_sizes, isNonIID = args.noniid)
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition,
                                                batch_size=args.bs,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=0)

        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        testset = torchvision.datasets.CIFAR100(root=args.datasetRoot,
                                            train=False,
                                            download=True,
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=args.bs,
                                                shuffle=False,
                                                num_workers=0)


    elif args.dataset == 'imagenet':
        datadir = args.datasetRoot
        traindir = os.path.join(datadir, 'CLS-LOC/train/')
        #valdir = os.path.join(datadir, 'CLS-LOC/')
        #testdir = os.path.join(datadir, 'CLS-LOC/')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.NotEvenPart:
            partition_sizes = np.random.randint(1,6, size=size)
            partition_sizes = args.datafrac*(partition_sizes/sum(partition_sizes))
            for _ in range(args.shuffleTimes):
                np.random.shuffle(partition_sizes)
        else:
            partition_sizes = [args.datafrac*(1.0 / size) for _ in range(size)]
        print("partition_sizes:{}".format(partition_sizes))
        partition = DataPartitioner(args, train_dataset, partition_sizes, isNonIID = args.noniid)
        partition = partition.use(rank)

        train_loader = torch.utils.data.DataLoader(
            partition, batch_size=args.bs, shuffle=True,
             pin_memory=True)
        '''
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.bs, shuffle=False,
            pin_memory=True)
        val_loader = None
        '''
        test_loader = None

    if args.dataset == 'emnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.EMNIST(root=args.datasetRoot,
                                              split = 'balanced',
                                              train=True,
                                              download=True,
                                              transform=transform_train)
        if args.NotEvenPart:
            partition_sizes = np.random.randint(1,6, size=size)
            partition_sizes = args.datafrac*(partition_sizes/sum(partition_sizes))
            for _ in range(args.shuffleTimes):
                np.random.shuffle(partition_sizes)
        else:
            partition_sizes = [args.datafrac*(1.0 / size) for _ in range(size)]
        print("partition_sizes:{}".format(partition_sizes))
        partition = DataPartitioner(args, train_dataset, partition_sizes, isNonIID=args.noniid)
        partition = partition.use(rank)

        train_loader = torch.utils.data.DataLoader(
            partition, batch_size=args.bs, shuffle=True,
             pin_memory=True)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.EMNIST(root=args.datasetRoot,
                                             split = 'balanced',
                                             train=False,
                                             download=True,
                                             transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=64,
                                                shuffle=False,
                                                num_workers=size)


    return train_loader, test_loader

def select_model(num_class, args):
    if args.model == 'VGG':
        model = vggnet.VGG(16, num_class)
    elif args.model == "googlenet":
        model = models.googlenet()
    elif args.model == "mobilenet_v2":
        model = models.mobilenet_v2()
    elif args.model == 'resnet':
        if args.dataset == 'cifar10' or "fashion-mnist":
            num_class = 10
            model = resnet.ResNet(50, num_class)
        elif args.dataset == "cifar100":
            num_class = 100
            model = resnet.ResNet(50, num_class)
        elif args.dataset == 'imagenet':
            model = models.resnet18()
    elif args.model == "resnet18":
        model = models.resnet18()
    elif args.model == 'wrn':
        if args.dataset == 'cifar10':
            num_class = 10
        elif args.dataset == "cifar100":
            num_class = 100
        model = wrn.Wide_ResNet(28,10,0,num_class)
    elif args.model == 'mlp':
        if args.dataset == 'emnist':
            model = MLP.MNIST_MLP(47)
    elif args.model == "resnet50":
        model = models.resnet50()
    elif args.model == "simpleFashion":
        model = simpleModel.FashionSimpleNet()
    elif args.model == "cnn2":
        model = simpleModel.CNN2()
    elif args.model == "scnn2":
        model = simpleModel.Net()
    if args.dataset == "fashion-mnist" and (args.model != "simpleFashion" and args.model!="cnn2" and args.model!="scnn2"):
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def select_graph_from_zoo(graphName):
    G = nx.Graph()
    parser = GraphMLParser()
    g = parser.parse("./archive/{}.graphml".format(graphName))
    for edge in g.edges():
        G.add_edges_from([(int(edge.node1.id), int(edge.node2.id))])
    return G

def select_graph_from_matlab(graphName):
    G = nx.Graph()
    g = loadgraph("./MatlabData/CAIDA_{}.mat".format(graphName))
    for edge in g.edges():
        G.add_edges_from([(int(edge.node1.id), int(edge.node2.id))])
    return G

def select_graph_from_roofnet(num_workers):
    G = nx.Graph()
    file = open("./avg_withcoordinates/pdr_1.txt", 'r')
    lines = file.readlines()
    for line in lines:
        words = line.strip().split(" ")
        G.add_edges_from([(int(words[0]), int(words[1]))])
    num_nodes = len(G.nodes)
    rm = num_nodes - num_workers
    if rm < 0 :
        print("need more workers")
        raise
    for i, degree in sorted(G.degree, key=itemgetter(1)):
        if i%4 == 0:
            continue
        if rm == 0:
            break
        else:
            G.remove_node(i)
            rm -= 1
    mapping = {}

    for i in range(len(sorted(G))):
        mapping[sorted(G)[i]] = i
    G = nx.relabel_nodes(G, mapping)
    return G



def select_graph_from(num_workers, f):
    G = nx.Graph()
    file = open("./{}".format(f), 'r')
    lines = file.readlines()
    for line in lines:
        words = line.strip().split(" ")
        n1 = int(words[0])
        n2 = int(words[1])
        if n1 == n2:
            #print("n1:{}, n2:{}".format(n1, n2))
            continue
        G.add_edges_from([(n1, n2)])
    num_nodes = len(G.nodes)
    rm = num_nodes - num_workers
    if rm < 0 :
        print("need more workers")
        raise
    #while rm>0:
    #print("rm:{}".format(rm))
    for i, degree in sorted(G.degree, key=itemgetter(1)):
        #if i%2 == 0:
        #    continue
        if rm == 0:
            break
        else:
            G.remove_node(i)
            rm -= 1
    #print("rm:{}".format(rm))
    mapping = {}
    for i in range(len(sorted(G))):
        mapping[sorted(G)[i]] = i
    G = nx.relabel_nodes(G, mapping)
    return G


def select_graph_from_roofnet_sp(num_workers, speed):
    G = nx.Graph()
    file = open("./avg_withcoordinates/pdr_{}.txt".format(speed), 'r')
    lines = file.readlines()
    for line in lines:
        words = line.strip().split(" ")
        G.add_edges_from([(int(words[0]), int(words[1]))])
    num_nodes = len(G.nodes)
    rm = num_nodes - num_workers
    if rm < 0 :
        print("need more workers")
        raise
    for i, degree in sorted(G.degree, key=itemgetter(1)):
        if i%4 == 0:
            continue
        if rm == 0:
            break
        else:
            G.remove_node(i)
            rm -= 1
    mapping = {}

    for i in range(len(sorted(G))):
        mapping[sorted(G)[i]] = i
    G = nx.relabel_nodes(G, mapping)
    return G

def select_graph_from_paper(graphName):
    Graphs =[
         #[[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
         # [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)],
         # [(2, 3), (2, 4), (2, 5), (2, 6), (2, 7)],
         # [(3, 4), (3, 5), (3, 6), (3, 7)],
         # [(4, 5), (4, 6), (4, 7)],
         # [(5, 6), (5, 7)],
         # [(6, 7)]],
         # graph 0:
         # 8-node erdos-renyi graph as shown in Fig. 1(a) in the main paper
         [[(1, 5), (6, 7), (0, 4), (2, 3)],
          [(1, 7), (3, 6)],
          [(1, 0), (3, 7), (5, 6)],
          [(1, 2), (7, 0)],
          [(3, 1)]],

         # graph 1:
         # 16-node gemetric graph as shown in Fig. A.3(a) in Appendix
         [[(4, 8), (6, 11), (7, 13), (0, 12), (5, 14), (10, 15), (2, 3), (1, 9)],
          [(11, 13), (14, 2), (5, 6), (15, 3), (10, 9)],
          [(11, 8), (2, 5), (13, 4), (14, 3), (0, 10)],
          [(11, 5), (15, 14), (13, 8)],
          [(2, 11)]],

         # graph 2:
         # 16-node gemetric graph as shown in Fig. A.3(b) in Appendix
         [[(2, 7), (12, 15), (3, 13), (5, 6), (8, 0), (9, 4), (11, 14), (1, 10)],
          [(8, 6), (0, 11), (3, 2), (5, 4), (15, 14), (1, 9)],
          [(8, 3), (0, 6), (11, 2), (4, 1), (12, 14)],
          [(8, 11), (6, 3), (0, 5)],
          [(8, 2), (0, 3), (6, 7), (11, 12)],
          [(8, 5), (6, 4), (0, 2), (11, 7)],
          [(8, 15), (3, 7), (0, 4), (6, 2)],
          [(8, 14), (5, 3), (11, 6), (0, 9)],
          [(8, 7), (15, 11), (2, 5), (4, 3), (1, 0), (13, 6)],
          [(12, 8)]],

         # graph 3:
         # 16-node gemetric graph as shown in Fig. A.3(c) in Appendix
         [[(3, 12), (4, 8), (1, 13), (5, 7), (9, 10), (11, 14), (6, 15), (0, 2)],
          [(7, 14), (2, 6), (5, 13), (8, 10), (1, 15), (0, 11), (3, 9), (4, 12)],
          [(2, 7), (3, 15), (9, 13), (6, 11), (4, 14), (10, 12), (1, 8), (0, 5)],
          [(5, 14), (1, 12), (13, 8), (9, 4), (2, 11), (7, 0)],
          [(5, 1), (14, 8), (13, 12), (10, 4), (6, 7)],
          [(5, 9), (14, 1), (13, 3), (8, 2), (11, 7)],
          [(5, 12), (14, 13), (1, 9), (8, 0)],
          [(5, 2), (14, 10), (1, 3), (9, 8), (13, 15)],
          [(5, 8), (14, 12), (1, 4), (13, 10)],
          [(5, 3), (14, 2), (9, 12), (1, 10), (13, 4)],
          [(5, 6), (14, 0), (8, 12), (1, 2)],
          [(5, 15), (9, 14)],
          [(11, 5)]],

         # graph 4:
         # 16-node erdos-renyi graph as shown in Fig 3.(b) in the main paper
         [[(2, 7), (3, 15), (13, 14), (8, 9), (1, 5), (0, 10), (6, 12), (4, 11)],
         [(12, 11), (5, 6), (14, 1), (9, 10), (15, 2), (8, 13)],
         [(12, 5), (11, 6), (1, 8), (9, 3), (2, 10)],
         [(12, 14), (11, 9), (5, 15), (0, 6), (1, 7)],
         [(12, 8), (5, 2), (11, 14), (1, 6)],
         [(12, 15), (13, 11), (10, 5), (3, 14)],
         [(12, 9)],
         [(0, 12)]],

         # graph 5, 8-node ring
         [[(0, 1), (2, 3), (4, 5), (6, 7)],
          [(0, 7), (2, 1), (4, 3), (6, 5)]],

         [[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
          [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)],
          [(2, 3), (2, 4), (2, 5), (2, 6), (2, 7)],
          [(3, 4), (3, 5), (3, 6), (3, 7)],
          [(4, 5), (4, 6), (4, 7)],
          [(5, 6), (5, 7)],
          [(6, 7)]]

        ]
    G = nx.Graph()
    for edges in Graphs[int(graphName)]:
        G.add_edges_from(edges)
    return G

def select_graph(graphid):
    # pre-defined base network topologies
    # you can add more by extending the list
    Graphs =[
             # graph 0:
             # 8-node erdos-renyi graph as shown in Fig. 1(a) in the main paper
             [[(1, 5), (6, 7), (0, 4), (2, 3)],
              [(1, 7), (3, 6)],
              [(1, 0), (3, 7), (5, 6)],
              [(1, 2), (7, 0)],
              [(3, 1)]],

             # graph 1:
             # 16-node gemetric graph as shown in Fig. A.3(a) in Appendix
             [[(4, 8), (6, 11), (7, 13), (0, 12), (5, 14), (10, 15), (2, 3), (1, 9)],
              [(11, 13), (14, 2), (5, 6), (15, 3), (10, 9)],
              [(11, 8), (2, 5), (13, 4), (14, 3), (0, 10)],
              [(11, 5), (15, 14), (13, 8)],
              [(2, 11)]],

             # graph 2:
             # 16-node gemetric graph as shown in Fig. A.3(b) in Appendix
             [[(2, 7), (12, 15), (3, 13), (5, 6), (8, 0), (9, 4), (11, 14), (1, 10)],
              [(8, 6), (0, 11), (3, 2), (5, 4), (15, 14), (1, 9)],
              [(8, 3), (0, 6), (11, 2), (4, 1), (12, 14)],
              [(8, 11), (6, 3), (0, 5)],
              [(8, 2), (0, 3), (6, 7), (11, 12)],
              [(8, 5), (6, 4), (0, 2), (11, 7)],
              [(8, 15), (3, 7), (0, 4), (6, 2)],
              [(8, 14), (5, 3), (11, 6), (0, 9)],
              [(8, 7), (15, 11), (2, 5), (4, 3), (1, 0), (13, 6)],
              [(12, 8)]],

             # graph 3:
             # 16-node gemetric graph as shown in Fig. A.3(c) in Appendix
             [[(3, 12), (4, 8), (1, 13), (5, 7), (9, 10), (11, 14), (6, 15), (0, 2)],
              [(7, 14), (2, 6), (5, 13), (8, 10), (1, 15), (0, 11), (3, 9), (4, 12)],
              [(2, 7), (3, 15), (9, 13), (6, 11), (4, 14), (10, 12), (1, 8), (0, 5)],
              [(5, 14), (1, 12), (13, 8), (9, 4), (2, 11), (7, 0)],
              [(5, 1), (14, 8), (13, 12), (10, 4), (6, 7)],
              [(5, 9), (14, 1), (13, 3), (8, 2), (11, 7)],
              [(5, 12), (14, 13), (1, 9), (8, 0)],
              [(5, 2), (14, 10), (1, 3), (9, 8), (13, 15)],
              [(5, 8), (14, 12), (1, 4), (13, 10)],
              [(5, 3), (14, 2), (9, 12), (1, 10), (13, 4)],
              [(5, 6), (14, 0), (8, 12), (1, 2)],
              [(5, 15), (9, 14)],
              [(11, 5)]],

             # graph 4:
             # 16-node erdos-renyi graph as shown in Fig 3.(b) in the main paper
             [[(2, 7), (3, 15), (13, 14), (8, 9), (1, 5), (0, 10), (6, 12), (4, 11)],
             [(12, 11), (5, 6), (14, 1), (9, 10), (15, 2), (8, 13)],
             [(12, 5), (11, 6), (1, 8), (9, 3), (2, 10)],
             [(12, 14), (11, 9), (5, 15), (0, 6), (1, 7)],
             [(12, 8), (5, 2), (11, 14), (1, 6)],
             [(12, 15), (13, 11), (10, 5), (3, 14)],
             [(12, 9)],
             [(0, 12)]],

             # graph 5, 8-node ring
             [[(0, 1), (2, 3), (4, 5), (6, 7)],
              [(0, 7), (2, 1), (4, 3), (6, 5)]]

            ]

    return Graphs[graphid]

def getGradLoss(model):
    grad_list = list()
    for para in model.parameters():
        grad_list.append(para.grad)
    flattened_grad = flatten_tensors(grad_list).cpu()
    return torch.norm(flattened_grad, p=2)**2

# INPUTS: output have shape of [batch_size, category_count]
#    and target in the shape of [batch_size] * there is only one true class for each sample
# topk is tuple of classes to be included in the precision
# topk have to a tuple so if you are giving one number, do not forget the comma
def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    #we do not need gradient calculation for those
    with torch.no_grad():
        #we will use biggest k, and calculate all precisions from 0 to k
        maxk = max(topk)
        batch_size = target.size(0)
        #topk gives biggest maxk values on dimth dimension from output
        #output was [batch_size, category_count], dim=1 so we will select biggest category scores for each batch
        # input=maxk, so we will select maxk number of classes
        #so result will be [batch_size,maxk]
        #topk returns a tuple (values, indexes) of results
        # we only need indexes(pred)
        _, pred = output.topk(maxk, 1, True, True)
        # then we transpose pred to be in shape of [maxk, batch_size]
        pred = pred.t()
        #we flatten target and then expand target to be like pred
        # target [batch_size] becomes [1,batch_size]
        # target [1,batch_size] expands to be [maxk, batch_size] by repeating same correct class answer maxk times.
        # when you compare pred (indexes) with expanded target, you get 'correct' matrix in the shape of  [maxk, batch_size] filled with 1 and 0 for correct and wrong class assignments
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        """ correct=([[0, 0, 1,  ..., 0, 0, 0],
        [1, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 1, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 1, 0,  ..., 0, 0, 0]], device='cuda:0', dtype=torch.uint8) """
        res = []
        # then we look for each k summing 1s in the correct matrix for first k element.
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Recorder(object):
    def __init__(self, args, rank):
        self.record_accuracy = list()
        self.record_timing = list()
        self.record_comp_timing = list()
        self.record_comm_timing = list()
        self.record_losses = list()
        self.record_trainacc = list()
        self.total_record_timing = list()
        self.sumActNeighbors = list()
        self.gradientloss = list()
        self.sumActTimes = list()
        self.rho = list()
        self.actLinks = list()
        self.actNodes = list()
        self.actMatchings = list()
        self.actBandwidth = list()
        self.args = args
        self.rank = rank
        self.saveFolderName = args.savePath + args.name + '_' + args.model
        #if rank == 0 and os.path.isdir(self.saveFolderName)==False and self.args.save:
        if rank == 0 and os.path.isdir(self.saveFolderName)==False:
            os.mkdir(self.saveFolderName)

    def add_new(self,record_time,comp_time,comm_time,epoch_time,top1,losses,test_acc, sumActNeighbors, sumActTimes, gradientloss, rho, actLinks, actMatchings, actBandwidth, actNodes):
        self.total_record_timing.append(record_time)
        self.record_timing.append(epoch_time)
        self.record_comp_timing.append(comp_time)
        self.record_comm_timing.append(comm_time)
        self.record_trainacc.append(top1)
        self.record_losses.append(losses)
        self.record_accuracy.append(test_acc)
        self.sumActNeighbors.append(sumActNeighbors)
        self.gradientloss.append(gradientloss)
        self.sumActTimes.append(sumActTimes)
        self.rho.append(rho)
        self.actLinks.append(actLinks)
        self.actNodes.append(actNodes)
        self.actMatchings.append(actMatchings)
        self.actBandwidth.append(actBandwidth)

    def save_to_file(self):
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-recordtime.log', self.total_record_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-time.log',  self.record_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-comptime.log',  self.record_comp_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-commtime.log',  self.record_comm_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-acc.log',  self.record_accuracy, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-losses.log',  self.record_losses, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-tacc.log',  self.record_trainacc, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-sumActNeighbors.log',  self.sumActNeighbors, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-gradientloss.log',  self.gradientloss, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-sumActTimes.log',  self.sumActTimes, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-rho.log',  self.rho, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-actLinks.log',  self.actLinks, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-actNodes.log',  self.actNodes, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-actMatchings.log',  self.actMatchings, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-noniid'+str(self.args.noniid)+'-asyn'+str(self.args.asyn)+'-gname'+str(self.args.graphname)+'-H'+str(self.args.H1)+'-'+str(self.args.H2)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-cost'+str(self.args.cost)+'-subg'+str(self.args.subgraph)+'-actBandwidth.log',  self.actBandwidth, delimiter=',')

        with open(self.saveFolderName+'/ExpDescription', 'w') as f:
            f.write(str(self.args)+ '\n')
            f.write(self.args.description + '\n')


def test(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    # correct = 0
    # total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        #inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        outputs = model(inputs)
        acc1 = comp_accuracy(outputs, targets)
        top1.update(acc1[0], inputs.size(0))
    return top1.avg
