import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import scipy.linalg
import gc

import faiss
import faiss.contrib.torch_utils

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret

class DiffusionNet(nn.Module):
    "Diffusion based decoder"

    def __init__(self, net_config):
        super(DiffusionNet, self).__init__()
        self.num_points = net_config['num_points']
        self.t_emb_dim=net_config['t_emb_dim']
        self.feature_dim=net_config['feature_dim']
        self.diffusion_mode = net_config.get('diffusion_mode', 'pointwise')
        self.input_dim = 3*self.num_points

        self.act = F.leaky_relu
        if self.diffusion_mode=='pointwise':
            self.decode_layers=nn.ModuleList([
                ConcatSquashLinear(3  , 128, self.feature_dim+self.t_emb_dim),
                ConcatSquashLinear(128, 256, self.feature_dim+self.t_emb_dim),
                ConcatSquashLinear(256, 512, self.feature_dim+self.t_emb_dim),
                ConcatSquashLinear(512, 256, self.feature_dim+self.t_emb_dim),
                ConcatSquashLinear(256, 128, self.feature_dim+self.t_emb_dim),
                ConcatSquashLinear(128,   3, self.feature_dim+self.t_emb_dim),
            ])
        elif self.diffusion_mode=='jointly':
            self.decode_layers=nn.ModuleList([
                ConcatSquashLinear(self.input_dim, 128, self.feature_dim+self.t_emb_dim),
                ConcatSquashLinear(128, 256, self.feature_dim+self.t_emb_dim),
                ConcatSquashLinear(256, 512, self.feature_dim+self.t_emb_dim),
                ConcatSquashLinear(512, 256, self.feature_dim+self.t_emb_dim),
                ConcatSquashLinear(256, 128, self.feature_dim+self.t_emb_dim),
                ConcatSquashLinear(128, self.input_dim, self.feature_dim+self.t_emb_dim),
            ])
        else:
            raise NotImplementedError(f"Unsupported diffusion mode {self.diffusion_mode}")
        
        self.time_enc= nn.Linear(3, self.t_emb_dim)

    def forward(self, x, t, feature):
        """
        Args:
            x : Noisy point cloud coords, (B, N, 3).
            t : Timestep, (B, ).
            feature : Encoded latent feature, (B, F).
        """
        batch_size, point_num, input_dim=x.shape
        t = t.view(batch_size, 1, 1)                # (B, 1, 1)
        feature = feature.view(batch_size, 1, -1)   # (B, 1, F)
        time_emb=torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1) # (B, 1, 3)
        time_emb = self.act(self.time_enc(time_emb))

        context=torch.cat([feature, time_emb], dim=-1) # (B, 1, F+t_emb_dim)

        if self.diffusion_mode == 'pointwise':
            out=x
        elif self.diffusion_mode == 'jointly':
            out=x.reshape(batch_size,-1)

        for i, layer in enumerate(self.decode_layers):
            out=layer(ctx=context, x=out)
            if i<len(self.decode_layers):
                out=self.act(out)
        
        if self.diffusion_mode=='pointwise':
            return out
        elif self.diffusion_mode=='jointly':
            return out.reshape(batch_size, point_num, input_dim)
    
class DiffusionPoints(nn.Module):
    def __init__(self, net_config, syntax):
        super().__init__()
        self.net = DiffusionNet(net_config)
        self.training_steps=net_config.get('training_steps', 100)
        self.beta_1 = net_config.get('beta_1', 1e-4)
        self.beta_T = net_config.get('beta_T', 0.05)   #use the timestep from https://arxiv.org/abs/2006.11239
        self.betas=torch.linspace(self.beta_1, self.beta_T, steps=self.training_steps).to('cuda')
        self.betas = torch.cat([torch.zeros([1]).to('cuda'), self.betas], dim=0) #padding
        self.alphas = 1-self.betas
        self.alpha_bars = torch.cumprod(self.alphas, 0)
        self.sigmas=torch.zeros_like(self.betas)
        for i in range(1, self.betas.shape[0]):
            self.sigmas[i]=((1-self.alpha_bars[i-1])/(1-self.alpha_bars[i]))*self.betas[i]
        self.sigmas = torch.sqrt(self.sigmas)
        self.init_method = net_config.get('init_method', 'gaussian')
        self.sample_radius = net_config.get('sample_radius', 1)
        self.num_points_fit = net_config.get('num_pooints_fit', 20)
        self.thres_dist = net_config.get('thres_dist', 10)

        self.faiss_resource, self.faiss_gpu_index_flat = None, None

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
        beta=self.betas[t]
        alpha_bar=self.alpha_bars[t]

        noise = torch.randn_like(x)
        c0 = torch.sqrt(alpha_bar).view(-1,1,1)    # (B,1,1)
        c1 = torch.sqrt(1-alpha_bar).view(-1,1,1)  # (B,1,1)
        noise_pred = self.net(c0*x+c1*noise, beta, feature)

        loss = F.mse_loss(noise_pred.view(-1, point_dim), noise.view(-1, point_dim), reduction='mean')
        
        return loss
    
    def noisy(self, x, t):
        beta = self.betas[t]
        alpha_bar = self.alpha_bars[t]
        noise = torch.randn_like(x)
        c0 = torch.sqrt(alpha_bar).view(-1,1,1)
        c1 = torch.sqrt(1-alpha_bar).view(-1,1,1)
        return c0*x + c1*noise
    
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
            print(f'step: {t}')
            alpha=self.alphas[t]
            alpha_bar=self.alpha_bars[t]
            z=torch.randn_like(x_T) if t>1 else torch.zeros_like(x_T)
            sigma = self.sigmas[t]

            c0=1.0/torch.sqrt(alpha)
            c1=(1-alpha)/torch.sqrt(1-alpha_bar)
            beta=self.betas[[t]*batch_size]
            x_t = traj[t]
            print(f'Debug: x_{t}: max{x_t.max()}, min{x_t.min()}')
            noise_pred=self.net(x_t, beta, feature)
            x_next=c0*(x_t - c1*noise_pred) + sigma*z
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
