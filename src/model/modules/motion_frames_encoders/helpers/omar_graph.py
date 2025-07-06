import math

import numpy as np

def conv_init(module):
    # he_normal
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A

def get_uniform_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = I - N
    return A

def get_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = np.stack((I, N))
    return A

def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_DAD_graph(num_node, self_link, neighbor):
    A = normalize_undigraph(edge2mat(neighbor + self_link, num_node))
    return A

def get_DLD_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    A = I - normalize_undigraph(edge2mat(neighbor, num_node))
    return A

class Graph():
    """
    The Graph to model the skeletons in NTU RGB+D 

    Arguments:
        labeling_mode: must be one of the follow candidates
            - uniform: Uniform Labeling
            - dastance*: Distance Partitioning*
            - dastance: Distance Partitioning
            - spatial: Spatial Configuration
            - DAD: normalized graph adjacency matrix
            - DLD: normalized graph laplacian matrix

    For more information, please refer to the section 'Partition Strategies' in our paper.
    """

    def __init__(self, config, labeling_mode='uniform'):
        # NOTE: labeling_mode is the strategy with which neighbors are defined in the graph to later perform convolution.
        # The purpose is to define how the neighborhood of a node v_ti is divided into a fixed number of partitions K and each partition
        # / subset will have its own learnable weight matrix.
        # kinematic_tree = [
        #     [0, 2, 5, 8, 11], # Left Leg,
        #     [0, 1, 4, 7, 10], # Right Leg 
        #     [0, 3, 6, 9, 12, 15], # Spine 
        #     [9, 14, 17, 19, 21], # Left Arm 
        #     [9, 13, 16, 18, 20], # Right Arm
        # ]
        
        if config == 'hml3d':
            num_node = 22
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [
                (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), # Spine
                (0, 2), (2, 5), (5, 8), (8, 11), # Left Leg
                (0, 1), (1, 4), (4, 7), (7, 10), # Right Leg
                (9, 13), (13, 16), (16, 18), (18, 20), # Left Arm
                (9, 14), (14, 17), (17, 19), (19, 21), # Right Arm
            ]
            
            inward = [(i, j) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            
            neighbor = inward + outward

        else:
            raise NotImplementedError

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.A = self.get_adjacency_matrix(num_node, self_link, neighbor, labeling_mode)

    def get_adjacency_matrix(self, num_node, self_link, neighbor, labeling_mode):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance*':
            A = get_uniform_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = get_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, self.inward, self.outward)
        elif labeling_mode == 'DAD':
            A = get_DAD_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'DLD':
            A = get_DLD_graph(num_node, self_link, neighbor)
        # elif labeling_mode == 'customer_mode':
        #     pass
        else:
            raise ValueError()
        return A