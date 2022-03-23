import numpy as np
import time
import torch
from mpi4py import MPI
from compressors import get_top_k

from comm_helpers import flatten_tensors, unflatten_tensors

class Communicator(object):
    """ Classs designed for communicating local models at workers """
    def __init__(self, rank, size):
        self.comm = MPI.COMM_WORLD
        self.rank = rank
        self.size = size

    def communicate(self, model):
        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # necessary preprocessing
        self.prepare_comm_buffer()

        # communication happens here
        # record the communication time
        comm_time = self.averaging()

        # Update local models
        self.reset_model()

        return comm_time

    def prepare_comm_buffer(self):
        raise NotImplemented

    def averaging(self):
        raise NotImplemented

    def reset_model(self):
        raise NotImplemented




class centralizedCommunicator(Communicator):
    """ Perform AllReduce at each iteration """
    def __init__(self, rank, size):
        super(centralizedCommunicator, self).__init__(rank, size)


    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()

    def averaging(self):
        self.comm.barrier()
        tic = time.time()

        # AllReduce
        self.recv_buffer = self.comm.allreduce(self.send_buffer, op=MPI.SUM)
        self.recv_buffer.div_(self.size)

        self.comm.barrier()
        toc = time.time()

        return toc - tic

    def reset_model(self):
        # Reset local models to be the averaged model
        #for f, t in zip(unflatten_tensors(self.recv_buffer.cuda(), self.tensor_list), self.tensor_list):
        for f, t in zip(unflatten_tensors(self.recv_buffer, self.tensor_list), self.tensor_list):
            t.set_(f)

class mixingMCommunicator(Communicator):
    def __init__(self, rank, size, topology, H1, H2):
        super(mixingMCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.iter = 1
        self.H1 = H1
        self.H2 = H2
        self.mixingMatrixW = topology.mixingMatrixW
        
    def prepare_comm_buffer(self):
        # faltten tensors
        param_data_list = list()
        for param in self.tensor_list:
            param_data_list.append(param.data)
        self.send_buffer = flatten_tensors(param_data_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)
    
    def averaging(self, active_flags):

        self.comm.barrier()
        tic = time.time()
        degree = 0 # record the degree of each node
        # decentralized averaging
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            else:
                if self.topology.neighbors_info[graph_id][self.rank] != []:
                    for neighbor_rank in self.topology.neighbors_info[graph_id][self.rank]:
                        
                        #if the weight between two nodes are 0, do not need to communicate at all.
                        if np.abs(self.mixingMatrixW[self.rank][neighbor_rank]) < 1e-6:
                            continue
                        
                        degree += 1
                        # Receive neighbor's model: x_j
                        self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                        # Aggregate neighbors' models: alpha * sum_j x_j
                        self.recv_buffer.add_(self.recv_tmp, alpha=self.mixingMatrixW[self.rank][neighbor_rank])
        # compute weighted average
        self.recv_buffer.add_(self.send_buffer, alpha=self.mixingMatrixW[self.rank][self.rank])
        self.comm.barrier()
        toc = time.time()
        return toc - tic, degree


    def reset_model(self):
        # Reset local models to be the averaged model
        #for f, t in zip(unflatten_tensors(self.recv_buffer.cuda(), self.tensor_list), self.tensor_list):
        for f, t in zip(unflatten_tensors(self.recv_buffer, self.tensor_list), self.tensor_list):
            with torch.no_grad():
                t.set_(f)


    def communicate(self, model):
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1
        #H=H1/H2
        if (self.iter-1)%self.H1 >= self.H2 :
            #print("skip the inter:{}".format(self.iter))
            return 0, 0
        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0, 0

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time, actNeighbors = self.averaging(active_flags)

        # update local models
        self.reset_model()

        return comm_time, actNeighbors
    
class decenCommunicator(Communicator):
    """ decentralized averaging according to a topology sequence """
    def __init__(self, rank, size, topology, H1, H2, asyn):
        super(decenCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.neighbor_weight = topology.neighbor_weight
        self.iter = 1
        self.H1 = H1
        self.H2 = H2
        self.asyn = asyn

    def prepare_comm_buffer(self):
        # faltten tensors
        param_data_list = list()
        for param in self.tensor_list:
            param_data_list.append(param.data)
        self.send_buffer = flatten_tensors(param_data_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)


    def averaging(self, active_flags):

        self.comm.barrier()
        tic = time.time()
        degree = 0 # record the degree of each node
        if not self.asyn:
            # decentralized averaging
            for graph_id, flag in enumerate(active_flags):
                if flag == 0:
                    continue
                else:
                    if self.topology.neighbors_info[graph_id][self.rank] != []:
                        for neighbor_rank in self.topology.neighbors_info[graph_id][self.rank]:
                            degree += 1
                            # Receive neighbor's model: x_j
                            self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                            # Aggregate neighbors' models: alpha * sum_j x_j
                            self.recv_buffer.add_(self.recv_tmp, alpha=self.neighbor_weight)

            # compute self weight according to degree
            selfweight = 1 - degree * self.neighbor_weight
            # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
            self.recv_buffer.add_(self.send_buffer, alpha=selfweight)
        else:
            numberSubgraph = len(active_flags)
            graph_id = np.random.randint(numberSubgraph)
            if self.topology.neighbors_info[graph_id][self.rank] != -1:
                degree += 1
                neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                self.recv_buffer.add_(self.recv_tmp, alpha=0.5)
                self.recv_buffer.add_(self.send_buffer, alpha=0.5)
        self.comm.barrier()
        toc = time.time()
        return toc - tic, degree


    def reset_model(self):
        # Reset local models to be the averaged model
        #for f, t in zip(unflatten_tensors(self.recv_buffer.cuda(), self.tensor_list), self.tensor_list):
        for f, t in zip(unflatten_tensors(self.recv_buffer, self.tensor_list), self.tensor_list):
            with torch.no_grad():
                t.set_(f)


    def communicate(self, model):
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1
        #H=H1/H2
        if (self.iter-1)%self.H1 >= self.H2 :
            #print("skip the inter:{}".format(self.iter))
            return 0, 0
        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0, 0

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time, actNeighbors = self.averaging(active_flags)

        # update local models
        self.reset_model()

        return comm_time, actNeighbors
    
    
    
class mixingMEachCommunicator(Communicator):
    def __init__(self, rank, size, topology, H1, H2):
        super(mixingMEachCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.iter = 1
        self.H1 = H1
        self.H2 = H2
        self.mixingMatrixWs = topology.mixingMatrixWs
        
    def prepare_comm_buffer(self):
        # faltten tensors
        param_data_list = list()
        for param in self.tensor_list:
            param_data_list.append(param.data)
        self.send_buffer = flatten_tensors(param_data_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)
    
    def averaging(self, active_flags):

        self.comm.barrier()
        tic = time.time()
        degree = 0 # record the degree of each node
        # decentralized averaging
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            else:
                if self.topology.neighbors_info[graph_id][self.rank] != []:
                    for neighbor_rank in self.topology.neighbors_info[graph_id][self.rank]:
                        
                        #if the weight between two nodes are 0, do not need to communicate at all.
                        if self.mixingMatrixWs[self.iter][self.rank][neighbor_rank] < 1e-10:
                            continue
                        
                        degree += 1
                        # Receive neighbor's model: x_j
                        self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                        # Aggregate neighbors' models: alpha * sum_j x_j
                        self.recv_buffer.add_(self.recv_tmp, alpha=self.mixingMatrixWs[self.iter][self.rank][neighbor_rank])
        # compute weighted average
        self.recv_buffer.add_(self.send_buffer, alpha=self.mixingMatrixWs[self.iter][self.rank][self.rank])
        self.comm.barrier()
        toc = time.time()
        return toc - tic, degree


    def reset_model(self):
        # Reset local models to be the averaged model
        #for f, t in zip(unflatten_tensors(self.recv_buffer.cuda(), self.tensor_list), self.tensor_list):
        for f, t in zip(unflatten_tensors(self.recv_buffer, self.tensor_list), self.tensor_list):
            with torch.no_grad():
                t.set_(f)


    def communicate(self, model):
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1
        #H=H1/H2
        if (self.iter-1)%self.H1 >= self.H2 :
            #print("skip the inter:{}".format(self.iter))
            return 0, 0
        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0, 0

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time, actNeighbors = self.averaging(active_flags)

        # update local models
        self.reset_model()

        return comm_time, actNeighbors
    
    
class mixingMRepeatCommunicator(Communicator):
    def __init__(self, rank, size, topology, H1, H2):
        super(mixingMRepeatCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.iter = 1
        self.H1 = H1
        self.H2 = H2
        self.mixingMatrixWs = topology.mixingMatrixWs
        self.top = topology.top
        
    def prepare_comm_buffer(self):
        # faltten tensors
        param_data_list = list()
        for param in self.tensor_list:
            param_data_list.append(param.data)
        self.send_buffer = flatten_tensors(param_data_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)
    
    def averaging(self, active_flags):

        self.comm.barrier()
        tic = time.time()
        degree = 0 # record the degree of each node
        # decentralized averaging
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            else:
                if self.topology.neighbors_info[graph_id][self.rank] != []:
                    for neighbor_rank in self.topology.neighbors_info[graph_id][self.rank]:
                        
                        #if the weight between two nodes are 0, do not need to communicate at all.
                        if self.mixingMatrixWs[self.iter%self.top][self.rank][neighbor_rank] < 1e-10:
                            continue
                        
                        degree += 1
                        # Receive neighbor's model: x_j
                        self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                        # Aggregate neighbors' models: alpha * sum_j x_j
                        self.recv_buffer.add_(self.recv_tmp, alpha=self.mixingMatrixWs[self.iter%self.top][self.rank][neighbor_rank])
        # compute weighted average                               
        self.recv_buffer.add_(self.send_buffer, alpha=self.mixingMatrixWs[self.iter%self.top][self.rank][self.rank])
        self.comm.barrier()
        toc = time.time()
        return toc - tic, degree


    def reset_model(self):
        # Reset local models to be the averaged model
        #for f, t in zip(unflatten_tensors(self.recv_buffer.cuda(), self.tensor_list), self.tensor_list):
        for f, t in zip(unflatten_tensors(self.recv_buffer, self.tensor_list), self.tensor_list):
            with torch.no_grad():
                t.set_(f)


    def communicate(self, model):
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1
        #H=H1/H2
        if (self.iter-1)%self.H1 >= self.H2 :
            #print("skip the inter:{}".format(self.iter))
            return 0, 0
        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0, 0

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time, actNeighbors = self.averaging(active_flags)

        # update local models
        self.reset_model()

        return comm_time, actNeighbors



class mixingMsCommunicator(Communicator):
    def __init__(self, rank, size, topology, H1, H2):
        super(mixingMsCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.iter = 1
        self.H1 = H1
        self.H2 = H2
        self.mixingMatrixWs = topology.mixingMatrixWs

    def prepare_comm_buffer(self):
        # faltten tensors
        param_data_list = list()
        for param in self.tensor_list:
            param_data_list.append(param.data)
        self.send_buffer = flatten_tensors(param_data_list).cpu()
        print("# model parameters:{}".format(len(self.send_buffer)))
        self.recv_buffer = torch.zeros_like(self.send_buffer)
    
    def averaging(self, active_flags):
        """
        activate only one topology each iteration
        """
        self.comm.barrier()
        tic = time.time()
        degree = 0 # record the degree of each node
        # decentralized averaging
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            else:
                activated_graph_id = graph_id
                break
        
        #if self.iter == 5:
        #print("self.iter:{}".format(self.iter))
        #print("rank:{}, activated_graph_id:{}".format(self.rank, activated_graph_id))
        #print("rank:{}, W:{}".format(self.rank, self.mixingMatrixWs[activated_graph_id]))
        
        self.recv_buffer.add_(self.send_buffer, alpha=self.mixingMatrixWs[activated_graph_id][self.rank][self.rank])
        

        #print("rank:{}, self-weight:{}".format(self.rank, self.mixingMatrixWs[activated_graph_id][self.rank][self.rank]))
        
        if self.topology.neighbors_info[activated_graph_id][self.rank] != []:
            
            #if self.iter == 5:
            #print("rank:{}, neighbors:{}".format(self.rank, self.topology.neighbors_info[activated_graph_id][self.rank]))
            
            for neighbor_rank in self.topology.neighbors_info[activated_graph_id][self.rank]:
                
                #if the weight between two nodes are 0, do not need to communicate at all.
                if self.mixingMatrixWs[activated_graph_id][self.rank][neighbor_rank] < 1e-10:
                    continue
                
                degree += 1
                # Receive neighbor's model: x_j
                self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                # Aggregate neighbors' models: alpha * sum_j x_j
                self.recv_buffer.add_(self.recv_tmp, alpha=self.mixingMatrixWs[activated_graph_id][self.rank][neighbor_rank])
                
                #if self.iter == 5:
                #print("rank:{}, neighbors:{}, weight:{}".format(self.rank, neighbor_rank, self.mixingMatrixWs[activated_graph_id][self.rank][neighbor_rank]))
        
        #if self.iter == 5:
        #print("degree:{}".format(degree))
        #raise
        self.comm.barrier()
        toc = time.time()
        return toc - tic, degree


    def reset_model(self):
        # Reset local models to be the averaged model
        #for f, t in zip(unflatten_tensors(self.recv_buffer.cuda(), self.tensor_list), self.tensor_list):
        for f, t in zip(unflatten_tensors(self.recv_buffer, self.tensor_list), self.tensor_list):
            with torch.no_grad():
                t.set_(f)


    def communicate(self, model):
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1
        #H=H1/H2
        if (self.iter-1)%self.H1 >= self.H2 :
            #print("skip the inter:{}".format(self.iter))
            return 0, 0
        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0, 0

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time, actNeighbors = self.averaging(active_flags)

        # update local models
        self.reset_model()

        return comm_time, actNeighbors
