import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.input_dim = 3*self.num_points

        self.act = F.leaky_relu
        self.decode_layers=nn.ModuleList([
            ConcatSquashLinear(3  , 128, self.feature_dim+self.t_emb_dim),
            ConcatSquashLinear(128, 256, self.feature_dim+self.t_emb_dim),
            ConcatSquashLinear(256, 512, self.feature_dim+self.t_emb_dim),
            ConcatSquashLinear(512, 256, self.feature_dim+self.t_emb_dim),
            ConcatSquashLinear(256, 128, self.feature_dim+self.t_emb_dim),
            ConcatSquashLinear(128,   3, self.feature_dim+self.t_emb_dim),
        ])
        self.time_enc=nn.ModuleList([
            nn.Linear(3, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 128),
            nn.Linear(128, self.t_emb_dim)
        ])

    def forward(self, x, t, feature):
        """
        Args:
            x : Noisy point cloud coords, (B, N, 3).
            t : Timestep, (B, ).
            feature : Encoded latent feature, (B, F).
        """
        batch_size=x.size(0)
        t = t.view(batch_size, 1, 1)                # (B, 1, 1)
        feature = feature.view(batch_size, 1, -1)   # (B, 1, F)
        time_emb=torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1) # (B, 1, 3)
        for i, layer in enumerate(self.time_enc):
            time_emb=layer(time_emb)
            time_emb=self.act(time_emb)

        context=torch.cat([feature, time_emb], dim=-1) # (B, 1, F+t_emb_dim)
        out=x
        for i, layer in enumerate(self.decode_layers):
            out=layer(ctx=context, x=out)
            if i<len(self.decode_layers):
                out=self.act(out)
        return out
    
class DiffusionPoints(nn.Module):
    def __init__(self, net_config, syntax):
        super().__init__()
        self.net = DiffusionNet(net_config)
        self.training_steps=net_config.get('training_steps', 100)
        self.beta_1 = net_config.get('beta_1', 1e-4)
        self.beta_T = net_config.get('beta_T', 0.05)   #use the timestep from https://arxiv.org/abs/2006.11239
        self.betas=torch.linspace(self.beta_1, self.beta_T, steps=self.training_steps).to('cuda')
        self.betas = torch.cat([torch.zeros([1]), self.betas], dim=0) #padding
        self.alphas = 1-self.betas
        self.alpha_bars = torch.cumprod(self.alphas, 0)
        self.sigmas=torch.zeros_like(self.betas)
        for i in range(1, self.betas.shape[0]):
            self.sigmas[i]=((1-self.alpha_bars[i-1])/(1-self.alpha_bars[i]))*self.betas[i]
        self.sigmas = torch.sqrt(self.sigmas)

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
    
    def sample(self, feature, return_traj = False):
        batch_size=feature.shape[0]
        x_T = torch.randn([batch_size, self.net.num_points, 3]).to(feature.device)
        traj = {self.training_steps: x_T}
        for t in range(self.training_steps, 0, -1):
            alpha=self.alphas[t]
            alpha_bar=self.alpha_bars[t]
            z=torch.randn_like(x_T) if t>1 else torch.zeros_like(x_T)
            sigma = self.sigmas[t]

            c0=1.0/torch.sqrt(alpha)
            c1=(1-alpha)/torch.sqrt(1-alpha_bar)
            beta=self.betas[[t]*batch_size]
            x_t = traj[t]
            noise_pred=self.net(x_t, beta, feature)
            x_next=c0*(x_t - c1*noise_pred) + sigma*z
            traj[t-1] = x_next.detach()
            traj[t] = traj[t].cpu()
            if not return_traj:
                del traj[t]
        if return_traj:
            return traj
        else:
            return traj[0]