import os
import numpy as np
import time
import argparse
import sys

from mpi4py import MPI
from math import ceil
from random import Random
import networkx as nx

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models
cudnn.benchmark = True
import sparsifier
import resnet
import vggnet
import wrn
import util
from graph_manager import FixedProcessor, MatchaProcessor, EnergeProcessor, FixedOptimalProcessor, FixedExactOptimalProcessor, BroadCastTopologyProcessor, FixedEachOptimalProcessor, MatchaExactProcessor, ResistanceSamplingProcessor, TopologySamplingProcessor, TopologySamplingPredictionProcessor
from communicator import decenCommunicator, centralizedCommunicator, mixingMCommunicator, mixingMEachCommunicator, mixingMRepeatCommunicator, mixingMsCommunicator
from datetime import datetime

def sync_allreduce(model, rank, size):
    senddata = {}
    recvdata = {}
    for param in model.parameters():
        tmp = param.data.cpu()
        senddata[param] = tmp.numpy()
        recvdata[param] = np.empty(senddata[param].shape, dtype = senddata[param].dtype)
    #torch.cuda.synchronize()
    comm.barrier()

    comm_start = time.time()
    for param in model.parameters():
        comm.Allreduce(senddata[param], recvdata[param], op=MPI.SUM)
    #torch.cuda.synchronize()
    comm.barrier()

    comm_end = time.time()
    comm_t = (comm_end - comm_start)

    for param in model.parameters():
        #param.data = torch.Tensor(recvdata[param]).cuda()
        param.data = torch.Tensor(recvdata[param])
        param.data = param.data/float(size)
    return comm_t

def run(rank, size):
    # set random seed
    #torch.manual_seed(args.randomSeed+rank)
    torch.manual_seed(args.randomSeed)
    np.random.seed(args.randomSeed)

    # init recorder
    sumActNeighbors, sumActTimes, actLinks, actMatchings, actBandwidth, actNodes = 0, 0, 0, 0, 0, 0
    rho = []
    recorder = util.Recorder(args,rank)
    losses = util.AverageMeter()
    gradientlosses = util.AverageMeter()
    top1 = util.AverageMeter()
    itr = 0
    
    # load data
    train_loader, test_loader = util.partition_dataset(rank, size, args)
    num_batches = ceil(len(train_loader.dataset) / float(args.bs))
    print("worker:{}, len(train_loader.dataset):{}, num_batchs:{}".format(rank, len(train_loader.dataset), num_batches))
    
    # if it is not traditional ML running on single machine, the network topology Graph and W need to be constructed
    if not args.single:
        # load base network topology
        if args.graphname == "roofnet":
            #Graph = util.select_graph_from_roofnet(size)
            Graph = util.select_graph_from_roofnet_sp(size, args.speed)
            args.graphname = "roofnet" + str(size) + args.speed
        elif args.graphname == "random":
            Graph = nx.gnm_random_graph(size, args.edges, seed=args.gSeed)
            args.graphname = "random" + str(size) + str(args.edges) + str(args.gSeed)
        elif args.graphname == "complete":
            Graph = sparsifier.get_complete_undirected_graph(size)
        else:
            Graph, bw = sparsifier.getGraphBandwidth(args.graphname, size, args.threshold)
        print("Graph info")
        print("#nodes:{}".format(len(Graph.nodes())))
        print("#edges:{}".format(len(Graph.edges())))
        
        if args.globalK:
            iterations = args.K
        else:
            iterations = args.epoch*num_batches
        
        # get the mixing matrix W from GP and deploy it in communicator
        if (args.cost=="time" or args.cost=="broadcast" or args.cost=="bandwidth") and (args.subgraph=="matchings" or args.subgraph=="links"):
            GP = MatchaProcessor(Graph, args.budget, rank, size, iterations, args.subgraph, args)
            communicator = decenCommunicator(rank, size, GP, args.H1, args.H2, args.asyn)

        elif args.cost=="energy":
            GP = EnergeProcessor(Graph, args.budget, rank, size, iterations, args.subgraph, args.ca, args.cb)
            communicator = decenCommunicator(rank, size, GP, args.H1, args.H2, args.asyn)

        elif args.cost=="fixedOptimal":
            GP = FixedOptimalProcessor(Graph, args.budget, rank, size, iterations, "links", args.ca, args.cb)
            communicator = decenCommunicator(rank, size, GP, args.H1, args.H2, args.asyn)

        elif args.cost=="fixedExactOptimal":
            GP = FixedExactOptimalProcessor(Graph, args.budget, rank, size, iterations, "links", args)
            communicator = mixingMCommunicator(rank, size, GP, args.H1, args.H2)

        elif args.cost=="topologyBroadcast":
            GP = BroadCastTopologyProcessor(Graph, args.budget, rank, size, iterations, args.subgraph, args.ca, args.cb)
            communicator = decenCommunicator(rank, size, GP, args.H1, args.H2, args.asyn)

        elif args.approach=="fixedEachOptimal":
            GP = FixedEachOptimalProcessor(Graph, args.budget, rank, size, iterations, "links", args)
            communicator = mixingMEachCommunicator(rank, size, GP, args.H1, args.H2)

        elif args.approach=="MatchaExact":
            GP = MatchaExactProcessor(Graph, args.budget, rank, size, iterations, "links")
            communicator = mixingMCommunicator(rank, size, GP, args.H1, args.H2)

        elif args.approach=="resistanceGS":
            GP = ResistanceSamplingProcessor(Graph, args.budget, rank, size, iterations, "links", args)
            communicator = mixingMRepeatCommunicator(rank, size, GP, args.H1, args.H2)

        elif args.approach=="topologySampling":
            GP = TopologySamplingProcessor(Graph, args.budget, rank, size, iterations, "resistGS", args, num_batches)
            communicator = mixingMsCommunicator(rank, size, GP, args.H1, args.H2)

        elif args.approach=="topologySamplingPrediction":
            GP = TopologySamplingPredictionProcessor(Graph, args.budget, rank, size, iterations, "allGreedyResistGS", args, num_batches)
            communicator = mixingMsCommunicator(rank, size, GP, args.H1, args.H2)

        elif args.approach=="fixedGreedy":
            GP = TopologySamplingPredictionProcessor(Graph, args.budget, rank, size, iterations, "fixedGreedy", args, num_batches)
            communicator = mixingMsCommunicator(rank, size, GP, args.H1, args.H2)

        elif args.approach=="fixedSparsifier":
            GP = TopologySamplingPredictionProcessor(Graph, args.budget, rank, size, iterations, "fixedSparsifier", args, num_batches)
            communicator = mixingMsCommunicator(rank, size, GP, args.H1, args.H2)

    # select neural network model
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset=="fashion-mnist":
        numClasses = 10
    model = util.select_model(numClasses, args)
    #model = model.cuda()
    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=0.0,
                          nesterov=args.nesterov)
    
    #if it continues running from previous jobs, it needs to load the previous result
    if args.initEpoch !=0:
        #checkpoint = torch.load("{}{}_{}/-gname{}-asyn{}-lr{}-H{}-{}-budget{}-r{}-cost{}-subg{}-noniid{}-noniidlevel{}".format(args.oldPath, args.name, args.model, args.graphname, args.asyn, args.lr, args.H1, args.H2, args.budget, rank, args.cost, args.subgraph, args.noniid, args.noniidlevel))
        checkpoint = torch.load("{}{}_{}/-gname{}-asyn{}-lr{}-H{}-{}-budget{}-r{}-cost{}-subg{}-noniid{}-noniidlevel{}epoch{}".format(args.oldPath, args.name, args.model, args.graphname, args.asyn, args.lr, args.H1, args.H2, args.budget, rank, args.cost, args.subgraph, args.noniid, args.noniidlevel, args.initEpoch))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = args.initEpoch
        if epoch != checkpoint["epoch"]:
            print('epoch != checkpoint["epoch"]')
            raise
        loss = checkpoint['loss']
    
    # guarantee all local models start from the same point
    # can be removed
    # sync_allreduce(model, rank, size)
    
    # k is used to count the iteration, when each worker has different size of local dataset
    k=1
    # start training
    for epoch in range(args.initEpoch, args.epoch):
        model.train()
        # Start training each epoch
        for batch_idx, (data, target) in enumerate(train_loader):

            # forward pass
            output = model(data)
            loss = criterion(output, target)

            #record training loss and accuracy
            acc1 = util.comp_accuracy(output, target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))

            # backward pass
            loss.backward()
            
            # if learning rate is not fixed, update it
            if not args.fixedLr:
                update_learning_rate(optimizer, epoch, itr=batch_idx, itr_per_epoch=len(train_loader))
            
            # gradient step
            optimizer.step()
            optimizer.zero_grad()
            
            # if it is not traditional ML running on single machine, workers need to communicate with each other
            if not args.single:
                # communication happens here
                actNeighbors = 0
                for i in range(args.rounds):
                    _, actNeighbors = communicator.communicate(model)
                
                # record information about the topology
                if actNeighbors > 0:
                    sumActTimes += 1
                actLinks += sparsifier.getNumActiveEdges(GP.activeTopology[itr])
                actNodes += sparsifier.getNumActiveNodes(GP.activeTopology[itr])
                actMatchings += sparsifier.getNumMatchings(GP.activeTopology[itr])
                if args.cost=="bandwidth":
                    actBandwidth += sparsifier.getBandwidth(GP.activeTopology[itr], bw)
                sumActNeighbors += actNeighbors
                rho.append(GP.rho[itr])
            
            itr += 1
            # when workers have different size, test_acc will be tested out every savePeriod times
            if args.globalK and k%args.savePeriod ==0:
                test_acc = util.test(model, test_loader)
                recorder.add_new(0,0,0,0,top1.avg,losses.avg, test_acc, sumActNeighbors, sumActTimes, gradientlosses.avg, np.average(rho), actLinks, actMatchings, actBandwidth, actNodes)
                recorder.save_to_file()

            k+=1
            if k >= args.K:
                break
        # if same size of local dataset, evaluate test accuracy at the end of each epoch
        if not args.globalK:
            test_acc = util.test(model, test_loader)
            recorder.add_new(0,0,0,0,top1.avg,losses.avg,test_acc, sumActNeighbors, sumActTimes, gradientlosses.avg, np.average(rho), actLinks, actMatchings, actBandwidth, actNodes)
            recorder.save_to_file()

        # reset recorders
        sumActNeighbors, sumActTimes, actLinks, actMatchings, actBandwidth, actNodes = 0, 0, 0, 0, 0, 0
        rho = []
        losses.reset()
        gradientlosses.reset()
        top1.reset()
        if k >= args.K:
            break
    recorder.save_to_file()

def update_learning_rate(optimizer, epoch, itr=None, itr_per_epoch=None,
                         scale=1):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    base_lr = 0.1
    target_lr = args.lr
    #lr_schedule = [30, 60]
    #lr_schedule = [50, 80, 100]#, 200]
    lr_schedule = [100, 150, 180, 200]
    lr = None
    if args.warmup and epoch < 5:  # warmup to scaled lr
        if target_lr <= base_lr:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - base_lr) * (count / (5 * itr_per_epoch))
            lr = base_lr + incr
    else:
        lr = target_lr
        for e in lr_schedule:
            if epoch >= e:
                lr *= 0.1

    if lr is not None:
        # print('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == "__main__":
    #mpiexec -np 13 python train_mpi.py --description "PyTorch CIFAR10 Training" --budget 0.5 --randomSeed 3 --datasetRoot "./data" --savePath "./recorder" --graphname "HiberniaCanada" --H1 1 --H2 1 --epoch 50 --cost "time" --subgraph "matchings"
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--name','-n', default="default", type=str, help='experiment name')
    parser.add_argument('--description', type=str, help='experiment description')
    parser.add_argument('--model', default="res", type=str, help='model name: res/VGG/wrn')
    parser.add_argument('--lr', default=0.8, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('--epoch', '-e', default=10, type=int, help='total epoch')
    parser.add_argument('--bs', default=64, type=int, help='batch size on each worker')
    parser.add_argument('--warmup', action='store_true', help='use lr warmup or not')
    parser.add_argument('--nesterov', action='store_true', help='use nesterov momentum or not')
    #parser.add_argument('--matcha', action='store_true', help='use MATCHA or not')
    parser.add_argument('--subgraph', type=str, help="links, matchings, or candidiates")
    parser.add_argument('--cost', type=str, help="time or energy")
    parser.add_argument('--budget', type=float, help='comm budget')
    parser.add_argument('--graphname', type=str, help='the idx of base graph')#Bics: 33/#BeyondTheNetwork 53
    parser.add_argument('--graphid', default=0, type=int, help='the idx of base graph')
    parser.add_argument('--H1', default=0, type=int, help='the period of update')
    parser.add_argument('--H2', default=0, type=int, help='the period of update')
    parser.add_argument('--dataset', default='cifar10', type=str, help='the dataset')
    parser.add_argument('--datasetRoot', type=str, help='the path of dataset')
    parser.add_argument('--p', '-p', action='store_true', help='partition the dataset or not')
    parser.add_argument('--savePath' ,type=str, help='save path')
    parser.add_argument('--compress', action='store_true', help='use chocoSGD or not')
    parser.add_argument('--consensus_lr', default=0.1, type=float, help='consensus_lr')
    parser.add_argument('--ca', default=4.7*0.8*0.25, type=float, help='computation cost')
    parser.add_argument('--cb', default=1.35*0.25, type=float, help='communication cost')
    parser.add_argument('--randomSeed', type=int, help='random seed')
    parser.add_argument('--asyn', action='store_true', help='syn or asyn')
    parser.add_argument('--noniid', action='store_true', help='iid or noniid')
    parser.add_argument('--testBudget', action='store_true', help='')
    parser.add_argument('--fixedLr', action='store_true', help='fixed lr')
    parser.add_argument('--single', action='store_true', help='')
    parser.add_argument('--error', type=str, help="")
    parser.add_argument('--print', type=str, help="")
    parser.add_argument('--alg', type=str, help="")
    parser.add_argument('--approach', type=str, help="")
    parser.add_argument('--noniidlevel', type=float, help='')
    parser.add_argument("--datafrac", default=1.0, type=float, help="")
    parser.add_argument("--epsilon", type=float, help="")
    parser.add_argument("--top", default=5, type=int, help="")
    parser.add_argument("--l", default=100000, type=int, help="")
    parser.add_argument("--low", type=float, help="")
    parser.add_argument("--high", type=float, help="")
    parser.add_argument("--speed", default="1", type=str, help="")
    parser.add_argument("--seq", action='store_true', help="")
    parser.add_argument("--broadcast", action='store_true', help="")
    parser.add_argument("--initEpoch", default=0, type=int, help="")
    parser.add_argument("--oldPath", default="", type=str, help="")
    parser.add_argument('--edges', default=0, type=int, help='')
    parser.add_argument('--gSeed', default=0, type=int, help='')
    parser.add_argument('--shuffleTimes', default=0, type=int, help='')
    parser.add_argument("--K", default=100000, type=int, help='')
    parser.add_argument("--savePeriod", default=100000, type=int, help="")
    parser.add_argument("--globalK", action='store_true', help="")
    parser.add_argument("--NotEvenPart", action='store_true', help="")
    parser.add_argument("--bStart", default=10, type=int, help='')
    parser.add_argument("--bEnd", default=100, type=int, help='')
    parser.add_argument("--bInterval", default=10, type=int, help='')
    parser.add_argument("--threshold", default=100000, type=int, help='')
    parser.add_argument('--heuristic_low', default=5, type=int, help='')
    parser.add_argument('--heuristic_high', default=100, type=int, help='')
    parser.add_argument('--rounds', default=1, type=int, help='')
    args = parser.parse_args()

    printpath = "./{}.txt".format(args.print)
    errorpath = "./{}.txt".format(args.error)
    sys.stdout = open(printpath, 'w')
    sys.stderr = open(errorpath, "w")

    if not args.description:
        print('No experiment description, exit!')
        exit()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    fromStart = datetime.now()
    run(rank, size)
    toEnd = datetime.now()
    print("Start Time =", fromStart.strftime("%H:%M:%S"))
    print("End Time =", toEnd.strftime("%H:%M:%S"))
    exit()
