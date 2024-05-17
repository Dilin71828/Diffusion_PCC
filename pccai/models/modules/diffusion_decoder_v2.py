import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import scipy.linalg
import gc

import faiss
import faiss.contrib.torch_utils

from pccai.models.modules.diffusion_decoder import DiffusionNet
from third_party.nndistance.modules.nnd import NNDModule

nndistance = NNDModule()
    
class DiffusionPointsV2(nn.Module):
    """
        Diffusion model that directly predict original point cloud from noisy one, by which
        some different loss functions can be used.
    """
    def __init__(self, net_config, syntax):
        super().__init__()
        self.net = DiffusionNet(net_config)
        self.training_steps=net_config.get('training_steps', 100)
        self.beta_1 = net_config.get('beta_1', 1e-4)
        self.beta_T = net_config.get('beta_T', 0.05)
        self.betas=torch.linspace(self.beta_1, self.beta_T, steps=self.training_steps).to('cuda')
        self.betas = torch.cat([torch.zeros([1]).to('cuda'), self.betas], dim=0) #padding
        self.index = torch.arange(self.training_steps+1, device ='cuda')
        self.alphas = 1-self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],(1,0),value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.one_minus_alphas_cumprod_prev = 1.0 - self.alphas_cumprod_prev
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1.0 - self.alphas_cumprod_prev)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0/self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_prev = torch.sqrt(1.0/self.alphas_cumprod_prev)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0/self.alphas_cumprod - 1)
        self.sigma = self.sqrt_one_minus_alphas_cumprod_prev/self.sqrt_one_minus_alphas_cumprod * torch.sqrt(1.0 - self.alphas_cumprod/self.alphas_cumprod_prev) 
        self.alpha_bars = torch.cumprod(self.alphas, 0)
        self.sigmas=torch.zeros_like(self.betas)
        for i in range(1, self.betas.shape[0]):
            self.sigmas[i]=((1-self.alpha_bars[i-1])/(1-self.alpha_bars[i]))*self.betas[i]
        self.sigmas = torch.sqrt(self.sigmas)
        self.init_method = net_config.get('init_method', 'gaussian')
        self.sample_radius = net_config.get('sample_radius', 1)
        self.num_points_fit = net_config.get('num_pooints_fit', 20)
        self.thres_dist = net_config.get('thres_dist', 10)
        self.loss_mode = net_config.get('loss_mode', 'MSE')

        self.faiss_resource, self.faiss_gpu_index_flat = None, None
    
    def noisy(self, x, t):
        noise = torch.randn_like(x)
        c0 = torch.sqrt(self.alphas_cumprod[t]).view(-1,1,1)
        c1 = torch.sqrt(self.one_minus_alphas_cumprod[t]).view(-1,1,1)
        return c0*x + c1*noise

    def get_loss(self, x, feature, t=None):
        """
        Args:
            x: Input residues, (B, N, 3)
            feature: Encoded feature vectors, (B, F)
        """
        batch_size, residule_size, point_dim=x.size()
        # Randomly sample t for each residule block
        if t==None:
            t=torch.randint(0, self.training_steps, (batch_size,), device=x.device)
        
        indexs = self.index[t]
        x_pred = self.net(self.noisy(x, t), indexs/self.training_steps, feature)
        weights = (self.alphas_cumprod[t]/self.one_minus_alphas_cumprod[t]).clamp(max=5)

        if self.loss_mode=='MSE':
            loss = F.mse_loss(x, x_pred, reduction='none').mean(dim=(1, 2))
            loss = (loss*weights).mean()
        elif self.loss_mode=='CD':
            dist_out,dist_x,_,_ = nndistance(x, x_pred)
            loss = ((dist_out.mean(dim=1)+dist_x.mean(dim=1))*weights).mean()
        
        return loss
    
    def sample(self, feature, return_traj = False, x_coarse = None, start_step = None, x_init = None):
        batch_size=feature.shape[0]
        if x_init==None:
            if self.init_method == 'gaussian':
                x_T = torch.randn([batch_size, self.net.num_points, 3]).to(feature.device)
            elif self.init_method == 'plane':
                # estimate normal using open3d
                pcd = o3d.t.geometry.PointCloud(x_coarse.cpu().numpy())
                pcd.estimate_normals(max_nn = self.num_points_fit, radius = self.thres_dist)
                normals = torch.tensor(pcd.point.normals.numpy(), device=feature.device) #[B, 3]
                tangents = normals.cross(torch.tensor([[0,0,1.]]*normals.shape[0]), dim=1)
                tangents_norm = tangents.norm(dim=1)
                tangents[tangents_norm<=1e-8] = torch.tensor([0,1.,0])
                tangents = tangents/tangents.norm(dim=1).reshape(-1,1).repeat(1,3)
                bitangents = normals.cross(tangents, dim=1)
                bitangents = bitangents/bitangents.norm(dim=1).reshape(-1,1).repeat(1,3)

                tangents = tangents.reshape(batch_size, 1, 3).repeat(1, self.net.num_points, 1)
                bitangents = bitangents.reshape(batch_size, 1, 3).repeat(1, self.net.num_points, 1)

                eps1 = (torch.rand([batch_size, self.net.num_points, 1])*np.pi*2).repeat(1, 1, 3)
                eps2 = (torch.rand([batch_size, self.net.num_points, 1])**(0.5)*self.sample_radius).repeat(1, 1, 3)
                x_T = torch.cos(eps1)*eps2*tangents + torch.sin(eps1)*eps2*bitangents
            elif self.init_method == 'quad':
                x_T = self.quad_fit(x_coarse = x_coarse, device=feature.device).detach()
        else:
            x_T = x_init

        gc.collect()
        torch.cuda.empty_cache()

        print('Start diffusion denoising...')
        if start_step == None:
            start_step = self.training_steps
        traj = {start_step: x_T}
        for t in range(start_step, 0, -1):
            indexs=self.index[[t]*batch_size]
            x_t = traj[t]
            x_pred = self.net(x_t, indexs/self.training_steps, feature)
            noise = (x_t - self.sqrt_alphas_cumprod[t]*x_pred)/self.sqrt_one_minus_alphas_cumprod[t]

            # DDIM sample
            x_next = self.sqrt_alphas_cumprod_prev[t]*x_pred + self.sqrt_one_minus_alphas_cumprod_prev[t]*noise

            print(f'Debug: x_{t}: max{x_t.max()}, min{x_t.min()}')
            traj[t-1] = x_next.detach()
            traj[t] = traj[t].cpu()
            if not return_traj:
                del traj[t]
            gc.collect()
            torch.cuda.empty_cache()
        if return_traj:
            return traj
        else:
            return traj[0]
    
    def quad_fit(self, x_coarse, device):
        batch_size = x_coarse.shape[0]
        # do nn searching on coarse pcd
        if self.faiss_gpu_index_flat == None:
            self.faiss_resource = faiss.StandardGpuResources()
            self.faiss_gpu_index_flat = faiss.GpuIndexFlatL2(self.faiss_resource, 3)
        self.faiss_gpu_index_flat.add(x_coarse)
        _, I = self.faiss_gpu_index_flat.search(x_coarse, self.num_points_fit)
        self.faiss_gpu_index_flat.reset()
        x_coarse_rep = x_coarse.unsqueeze(1).repeat_interleave(self.num_points_fit, dim=1)
        neighbors = x_coarse[I] - x_coarse_rep
        # remove outliers
        mask = torch.logical_or(
            torch.max(neighbors, dim=2)[0] > self.thres_dist,
            torch.min(neighbors, dim=2)[0] < -self.thres_dist
        )
        I[mask] = I[:,0].unsqueeze(-1).repeat_interleave(self.num_points_fit, dim=1)[mask]
        neighbors[mask] = x_coarse[I[mask]] - x_coarse_rep[mask]
        # fit quad surface
        # select the 'most flat' axis
        expand = neighbors.max(dim=1)[0] - neighbors.min(dim=1)[0]
        axis_index = expand.argsort(dim=1, descending = True).reshape(-1,1,3).repeat(1, self.num_points_fit, 1)
        coord_x = neighbors.gather(dim=2, index=axis_index[:, :, :1])
        coord_y = neighbors.gather(dim=2, index=axis_index[:,:,1:2])
        coord_z = neighbors.gather(dim=2, index=axis_index[:,:,2:])
        A = torch.cat([torch.ones((neighbors.shape[0],self.num_points_fit,1),device=device),
                       coord_x, coord_y, coord_x**2, coord_y**2, coord_x*coord_y],dim=2)
        params = torch.linalg.lstsq(A, coord_z)[0]
        normals = -torch.cat([params[:,1], params[:,2], -torch.ones((neighbors.shape[0],1),device=device)], dim=1) #calculate normal at x=0, y=0
        #sample data in disk area
        eps1 = torch.rand([batch_size, self.net.num_points, 1], device=device)*np.pi*2
        eps2 = torch.rand([batch_size, self.net.num_points, 1], device=device)**(0.5)*self.sample_radius
        sample_x = torch.cos(eps1)*eps2
        sample_y = torch.sin(eps1)*eps2
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
            torch.einsum('ijk,ik->ij',sample_coord,tangents).reshape(-1,self.net.num_points,1),
            torch.einsum('ijk,ik->ij',sample_coord,bitangents).reshape(-1,self.net.num_points,1),
            torch.einsum('ijk,ik->ij',sample_coord,normals).reshape(-1,self.net.num_points,1)
        ], dim=2)
        sample_x = sample_coord[:,:,0].reshape(sample_coord.shape[0],self.net.num_points,1)
        sample_y = sample_coord[:,:,1].reshape(sample_coord.shape[0],self.net.num_points,1)
        sample_quad_coord = torch.cat([torch.ones((neighbors.shape[0],self.net.num_points,1),device='cuda'),
                                       sample_x,sample_y,sample_x**2,sample_y**2,sample_x*sample_y],dim=2)
        sample_coord[:,:,2] = torch.einsum('ijk,ik->ij',sample_quad_coord, params.squeeze(2))
        x_T = sample_coord.scatter(dim=2,index=axis_index[:,::self.num_points_fit,:].repeat(1, self.net.num_points,1),src=sample_coord)
        x_T = x_T.clamp(-self.sample_radius*2, self.sample_radius*2)
        return x_T
