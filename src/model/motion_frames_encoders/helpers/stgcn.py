import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Graph():
    """
    The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(
        self,
        layout='openpose',
        strategy='uniform',
        max_hop=1,
        dilation=1
    ):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'hml3d':
            self.num_node = 22
            neighbor_link = [
                (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), # Spine
                (0, 2), (2, 5), (5, 8), (8, 11), # Left Leg
                (0, 1), (1, 4), (4, 7), (7, 10), # Right Leg
                (9, 13), (13, 16), (16, 18), (18, 20), # Left Arm
                (9, 14), (14, 17), (17, 19), (19, 21), # Right Arm
            ]
            self_link = [(i, i) for i in range(22)]
            self.edge = self_link + neighbor_link
            self.center = 9
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


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

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):
    r"""
    The basic module for applying a graph convolution.
    It is designed to perform a Temporal and Spatial convolution over a graph sequence in a coupled manner (simultaneously).
    Its main goal is to learn the spatial and temporal features of the input data.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            # NOTE: out_channels * kernel_size; This way we have an output channel for each partition of the K (kernel_size).
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        # NOTE: we apply the 2D convolution (temporal convolution + feature transformation).
        x = self.conv(x)
        # (N, out_channels * K, T_out, V); T_out depend on t_stride and t_padding.

        # NOTE: n=N, kc=out_channels*K, t=T_out, v=V
        n, kc, t, v = x.size()
        
        # NOTE: reshape and prepare for spatial graph convolution
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        # kc // self.kernel_size effectively gives out_channels
        # (N, K, out_channels, T_out, V)
        # this separates the 'out_channels * K' into K groups, each with 'out_channels'; each group of channels will correspond to one partition of A.
        
        # NOTE: perform spatial Graph Convolution using Einstein summation
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        # (N, out_channels, T_out, V)

        return x.contiguous(), A

import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCN(nn.Module):
    r"""
    Spatial temporal graph convolutional networks.
    Its goal is to simultaneously learn the spatial and temporal features / patterns of the input data.
    This learning need to be done automatically (end-to-end) without manual subdivision of body parts or temporal segments.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(
        self,
        in_channels,
        num_class,
        graph_args,
        edge_importance_weighting,
        latent_dim,
        **kwargs
    ):
        super().__init__()

        # NOTE: load graph, it is a static one that describes the human body skeleton and how the joints are connected.
        self.graph = Graph(**graph_args)
        
        # NOTE: required to define neighborhoods on the graph adn thus guide spatial convolution.
        # 
        ADJACENCY_MATRIX = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', ADJACENCY_MATRIX)

        # build networks
        spatial_kernel_size = ADJACENCY_MATRIX.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        # NOTE: batch normalization
        self.data_bn = nn.BatchNorm1d(in_channels * ADJACENCY_MATRIX.size(1))
        
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        
        # NOTE: ST-GCN layers
        # At each layer, we apply separately a spatial and then a temporal convolution, convolutions aren't done simultaneously / jointly.
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))
        
        self.latent_dim = latent_dim

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # NOTE: classification layer
        # take the 256 channels from the last st_gcn layer and map to num_class by applying a 1x1 convolution
        self.fcn = nn.Conv2d(256, latent_dim, kernel_size=1)

    def forward(self, x):

        N, C, T, V, M = x.size()
        # N = batch size
        # C = number of channels
        # T = length of input sequence
        # V = number of graph nodes
        # M = number of persons in a frame
        
        # --- --- --- --- --- --- --- --- --- --- --- ---
        
        # NOTE: data normalization
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        
        # --- --- --- --- --- --- --- --- --- --- --- ---

        # NOTE: forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        # NOTE: (N*M, 256, T_final, V)
        
        # --- --- --- --- --- --- --- --- --- --- --- ---

        # NOTE: global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        # (N*M, 256, 1, 1)
        # NOTE: if multiple persons (M > 1), average their features
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        # (N, 256, 1, 1)

        # --- --- --- --- --- --- --- --- --- --- --- ---

        # NOTE: prediction, apply 1x1 Conv2d: (N, num_class, 1, 1)
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        # (N, num_class)
        
        # --- --- --- --- --- --- --- --- --- --- --- ---

        return x

    def encode(self, x):
        N, C, T, V, M = x.size()
        # N: batch_size
        # C: in_channels
        # T: time_frames
        # V: num_joints
        # M: num_persons
        
        # --- --- --- --- --- --- --- --- --- --- --- ---
        
        # NOTE: data normalization
        x = x.permute(0, 4, 3, 1, 2).contiguous() # (N, M, V, C, T)
        x = x.view(N * M, V * C, T) # (N*M, V*C, T) for BatchNorm1d
        
        x = self.data_bn(x)
        
        x = x.view(N, M, V, C, T) # Reshape back
        x = x.permute(0, 1, 3, 4, 2).contiguous() # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)
        # NOTE: (N*M, C, T, V); final shape for st_gcn blocks
        
        # --- --- --- --- --- --- --- --- --- --- --- ---

        # NOTE: forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
            
        # NOTE: after the loop, x has shape: (N*M, 256, T_final, V); with T_final = T / 4 (due to two layers with stride=2 in self.st_gcn_networks)
        
        _, c, t, v = x.size()
        # NOTE: reshape x to separate N and M, and permute to a common "feature" format
        # (N, 256, T_final, V, M)
        # raw feature map output by the last ST-GCN layer, per person.

        # --- --- --- --- --- --- --- --- --- --- --- ---
        
        # NOTE: global pooling on the spatial (nodes) and time dimension
        x = F.avg_pool2d(x, x.size()[2:])
        # (N*M, 256, 1, 1)
        # NOTE: if multiple persons (M > 1), average their features
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        # (N, 256, 1, 1)
        
        # --- --- --- --- --- --- --- --- --- --- --- ---
        
        x = x.view(x.size(0), -1)
        # (N, 256)
        
        # --- --- --- --- --- --- --- --- --- --- --- ---
        
        return x

    def extract_feature(self, x):

        N, C, T, V, M = x.size()
        # N: batch_size
        # C: in_channels
        # T: time_frames
        # V: num_joints
        # M: num_persons
        
        # --- --- --- --- --- --- --- --- --- --- --- ---
        
        # NOTE: data normalization
        x = x.permute(0, 4, 3, 1, 2).contiguous() # (N, M, V, C, T)
        x = x.view(N * M, V * C, T) # (N*M, V*C, T) for BatchNorm1d
        
        x = self.data_bn(x)
        
        x = x.view(N, M, V, C, T) # Reshape back
        x = x.permute(0, 1, 3, 4, 2).contiguous() # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)
        # NOTE: (N*M, C, T, V); final shape for st_gcn blocks
        
        # --- --- --- --- --- --- --- --- --- --- --- ---

        # NOTE: forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
            
        # NOTE: after the loop, x has shape: (N*M, 256, T_final, V); with T_final = T / 4 (due to two layers with stride=2 in self.st_gcn_networks)
        
        _, c, t, v = x.size()
        # NOTE: reshape x to separate N and M, and permute to a common "feature" format
        # (N, 256, T_final, V, M)
        # raw feature map output by the last ST-GCN layer, per person.
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # --- --- --- --- --- --- --- --- --- --- --- ---
        
        # NOTE: prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        # --- --- --- --- --- --- --- --- --- --- --- ---
        
        return output, feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(
        self,
        # NOTE: number of input channels / features per node
        in_channels,
        # NOTE: number of output channels / features per node
        out_channels,
        # NOTE: (temporal_kernel_size, spatial_kernel_size_K)
        kernel_size,
        # NOTE: stride for temporal convolution
        stride=1,
        dropout=0,
        residual=True
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        
        padding = ((kernel_size[0] - 1) // 2, 0)

        # NOTE: slight temporal effect as temporal kernel size here is 1, mainly for spatial / graph convolution
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        # NOTE: temporal convolution
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # NOTE: 1D temporal convolution
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1), # Kernel (TemporalDim, 1_SpatialDim)
                (stride, 1),         # Stride (TemporalStride, 1_SpatialStride)
                padding,             # Temporal padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        # x: input features (N*M, C_in, T_in, V)
        # A: adjacency matrix (K_spatial, V, V)
        
        res = self.residual(x)
        
        x, A = self.gcn(x, A)
        # NOTE: (N*M, C_out, T_in, V)
        
        x = self.tcn(x) + res
        # NOTE: (N*M, C_out, T_out, V); T_out depends on stride

        x = self.relu(x)
        
        return x, A