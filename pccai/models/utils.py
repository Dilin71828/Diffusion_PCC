# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Elementary modules and utility functions to process point clouds

import numpy as np
import torch
import torch.nn as nn

from pccai.utils.misc import sample_x_10, sample_y_10

import faiss
import faiss.contrib.torch_utils


def get_Conv2d_layer(dims, kernel_size, stride, doLastRelu):
    """Elementary 2D convolution layers."""

    layers = []
    for i in range(1, len(dims)):
        padding = int((kernel_size - 1) / 2) if kernel_size != 1 else 0
        layers.append(nn.Conv2d(in_channels=dims[i-1], out_channels=dims[i],
            kernel_size=kernel_size, stride=stride, padding=padding, bias=True))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU(inplace=True))
    return layers # nn.Sequential(*layers)


class Conv2dLayers(nn.Sequential):
    """2D convolutional layers.

    Args:
        dims: dimensions of the channels
        kernel_size: kernel size of the convolutional layers.
        doLastRelu: do the last Relu (nonlinear activation) or not.
    """
    def __init__(self, dims, kernel_size, doLastRelu=False):
        layers = get_Conv2d_layer(dims, kernel_size, 1, doLastRelu) # Note: may need to init the weights and biases here
        super(Conv2dLayers, self).__init__(*layers)


def get_and_init_FC_layer(din, dout, init_bias='zeros'):
    """Get a fully-connected layer."""

    li = nn.Linear(din, dout)
    #init weights/bias
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    if init_bias == 'uniform':
        nn.init.uniform_(li.bias)
    elif init_bias == 'zeros':
        li.bias.data.fill_(0.)
    else:
        raise 'Unknown init ' + init_bias
    return li


def get_MLP_layers(dims, doLastRelu, init_bias='zeros'):
    """Get a series of MLP layers."""

    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i-1], dims[i], init_bias=init_bias))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class PointwiseMLP(nn.Sequential):
    """PointwiseMLP layers.

    Args:
        dims: dimensions of the channels
        doLastRelu: do the last Relu (nonlinear activation) or not.
        Nxdin ->Nxd1->Nxd2->...-> Nxdout
    """
    def __init__(self, dims, doLastRelu=False, init_bias='zeros'):
        layers = get_MLP_layers(dims, doLastRelu, init_bias)
        super(PointwiseMLP, self).__init__(*layers)


class GlobalPool(nn.Module):
    """BxNxK -> BxK"""

    def __init__(self, pool_layer):
        super(GlobalPool, self).__init__()
        self.Pool = pool_layer

    def forward(self, X):
        X = X.unsqueeze(-3) #Bx1xNxK
        X = self.Pool(X)
        X = X.squeeze(-2)
        X = X.squeeze(-2)   #BxK
        return X


class PointNetGlobalMax(nn.Sequential):
    """BxNxdims[0] -> Bxdims[-1]"""

    def __init__(self, dims, doLastRelu=False):
        layers = [
            PointwiseMLP(dims, doLastRelu=doLastRelu),      #BxNxK
            GlobalPool(nn.AdaptiveMaxPool2d((1, dims[-1]))),#BxK
        ]
        super(PointNetGlobalMax, self).__init__(*layers)


class PointNetGlobalAvg(nn.Sequential):
    """BxNxdims[0] -> Bxdims[-1]"""

    def __init__(self, dims, doLastRelu=True):
        layers = [
            PointwiseMLP(dims, doLastRelu=doLastRelu),      #BxNxK
            GlobalPool(nn.AdaptiveAvgPool2d((1, dims[-1]))),#BxK
        ]
        super(PointNetGlobalAvg, self).__init__(*layers)


class PointNet(nn.Sequential):
    """Vanilla PointNet Model.

    Args:
        MLP_dims: dimensions of the pointwise MLP
        FC_dims: dimensions of the FC to process the max pooled feature
        doLastRelu: do the last Relu (nonlinear activation) or not.
        Nxdin ->Nxd1->Nxd2->...-> Nxdout
    """
    def __init__(self, MLP_dims, FC_dims, MLP_doLastRelu):
        assert(MLP_dims[-1]==FC_dims[0])
        layers = [
            PointNetGlobalMax(MLP_dims, doLastRelu=MLP_doLastRelu),#BxK
        ]
        layers.extend(get_MLP_layers(FC_dims, False))
        super(PointNet, self).__init__(*layers)

def quad_fitting(x_coarse, fit_num = 50, fit_radius = 20, sample_mode = 'random', sample_num=10, sample_radius=2):
    """
    Fit a quadratic surface for each points in a coarse point cloud based on its neighbor, then
    sample points from a local neighbor area.

    Args:
        x_coarse: positions of point, [B, N, Dim]
        fit_num: number of neibor points used to fit quadratic surface
        fit_radius: restrict the maximum distance between the neighbor points and the center
        sample_mode: indicate different ways to generate samples ('random' for uniform sampling, 'predefined' will use some fixed pattern)
        sample_num: number of output points
        sample_radius: control the size of the sampled area
    """

    batch_size = x_coarse.shape[0]
    faiss_resource = faiss.StandardGpuResources()
    faiss_gpu_index_flat = faiss.GpuIndexFlatL2(faiss_resource, 3)
    faiss_gpu_index_flat.add(x_coarse)
    _, I = faiss_gpu_index_flat.search(x_coarse, fit_num)
    faiss_gpu_index_flat.reset()
    x_coarse_rep = x_coarse.unsqueeze(1).repeat_interleave(fit_num, dim=1)
    neighbors = x_coarse[I] - x_coarse_rep
    # remove outliers
    mask = torch.logical_or(
        torch.max(neighbors, dim=2)[0] > fit_radius,
        torch.min(neighbors, dim=2)[0] < -fit_radius
    )
    I[mask] = I[:,0].unsqueeze(-1).repeat_interleave(fit_num, dim=1)[mask]
    neighbors[mask] = x_coarse[I[mask]] - x_coarse_rep[mask]
    # fit quad surface
    # select the 'most flat' axis
    expand = neighbors.max(dim=1)[0] - neighbors.min(dim=1)[0]
    axis_index = expand.argsort(dim=1, descending = True).reshape(-1,1,3).repeat(1, fit_num, 1)
    coord_x = neighbors.gather(dim=2, index=axis_index[:, :, :1])
    coord_y = neighbors.gather(dim=2, index=axis_index[:,:,1:2])
    coord_z = neighbors.gather(dim=2, index=axis_index[:,:,2:])
    A = torch.cat([torch.ones((neighbors.shape[0],fit_num,1),device=x_coarse.device),
                   coord_x, coord_y, coord_x**2, coord_y**2, coord_x*coord_y],dim=2)
    params = torch.linalg.lstsq(A, coord_z)[0]
    normals = -torch.cat([params[:,1], params[:,2], -torch.ones((neighbors.shape[0],1),device=x_coarse.device)], dim=1) #calculate normal at x=0, y=0
    #sample data in disk area
    if sample_mode=='random':
        eps1 = torch.rand([batch_size, sample_num, 1], device=x_coarse.device)*np.pi*2
        eps2 = torch.rand([batch_size, sample_num, 1], device=x_coarse.device)**(0.5)*sample_radius
        sample_x = torch.cos(eps1)*eps2
        sample_y = torch.sin(eps1)*eps2
    elif sample_mode=='predefined':
        if sample_num==10:
            sample_x = torch.from_numpy(sample_x_10, device=x_coarse.device).reshape(1, -1, 1).repeat(batch_size, 1, 1)*sample_radius
            sample_y = torch.from_numpy(sample_y_10, device=x_coarse.device).reahspe(1, -1, 1).repeat(batch_size, 1, 1)*sample_radius
        else:
            raise NotImplementedError(f'The pattern to sample {sample_num} points is not defined.')
        pass
    else:
        raise NotImplementedError(f'sample mode {sample_mode} not supported!')
    # transform to world space
    normals = normals/normals.norm(dim=1).reshape(-1,1).repeat(1,3)
    tangents = normals.cross(torch.tensor([[0,0,1.]]*normals.shape[0],device='cuda'),dim=1)
    tangents_norm = tangents.norm(dim=1)
    tangents[tangents_norm<1e-8] = torch.tensor([0,1.,0],device='cuda')
    tangents = tangents/tangents.norm(dim=1).reshape(-1,1).repeat(1,3)  # [B,3]
    bitangents = normals.cross(tangents,dim=1)
    bitangents = bitangents/bitangents.norm(dim=1).reshape(-1,1).repeat(1,3) # [B,3]
    sample_coord = torch.cat([sample_x,sample_y,torch.zeros_like(sample_x)],dim=2)
    sample_coord = torch.cat([
        torch.einsum('ijk,ik->ij',sample_coord,tangents).reshape(-1,sample_num,1),
        torch.einsum('ijk,ik->ij',sample_coord,bitangents).reshape(-1,sample_num,1),
        torch.einsum('ijk,ik->ij',sample_coord,normals).reshape(-1,sample_num,1)
    ], dim=2)
    sample_x = sample_coord[:,:,0].reshape(sample_coord.shape[0],sample_num,1)
    sample_y = sample_coord[:,:,1].reshape(sample_coord.shape[0],sample_num,1)
    sample_quad_coord = torch.cat([torch.ones((neighbors.shape[0],sample_num,1),device=x_coarse.device),
                                   sample_x,sample_y,sample_x**2,sample_y**2,sample_x*sample_y],dim=2)
    sample_coord[:,:,2] = torch.einsum('ijk,ik->ij',sample_quad_coord, params.squeeze(2))
    sample_coord = sample_coord.scatter(dim=2,index=axis_index[:,::fit_num,:].repeat(1, sample_num,1),src=sample_coord)
    sample_coord = sample_coord.clamp(-sample_radius*2, sample_radius*2)
    return sample_coord