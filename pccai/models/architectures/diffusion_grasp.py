import os, sys
import torch
import torch.nn as nn
import numpy as np
import time
import MinkowskiEngine as ME

from pccai.models.modules.get_modules import get_module_class
from pccai.models.utils_sparse import scale_sparse_tensor_batch, sort_sparse_tensor_with_dir

#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../third_party/PCGCv2'))
from third_party.PCGCv2.entropy_model import EntropyBottleneck

class DiffusionGeoResCompression(nn.Module):

    def __init__(self, net_config, syntax):
        super(DiffusionGeoResCompression, self).__init__()

        self.dus = net_config.get('dus',1)
        self.scaling_ratio = net_config['scaling_ratio']
        self.eb_channel = net_config['entropy_bottleneck']
        self.entropy_bottleneck = EntropyBottleneck(self.eb_channel)
        self.thres_dist = np.ceil((1/self.scaling_ratio)*0.65) if self.scaling_ratio < 0.5 else 1

        self.point_mul = net_config.get('point_mul', 5)
        self.skip_mode = net_config.get('skip_mode', False)

        if syntax.phase.lower() == 'train':
            self.noise=net_config.get('noise', -1)
        
        net_config['res_enc']['k'] =  net_config['res_dec']['num_points'] = self.point_mul
        net_config['res_enc']['thres_dist'] = self.thres_dist
        net_config['res_dec']['dims'][0] = net_config['vox_dec']['dims'][-1]

        self.res_dec = get_module_class(net_config['res_dec']['model'], False)(net_config['res_dec'], syntax=syntax)
        self.vox_dec = get_module_class(net_config['vox_dec']['model'], False)(net_config['vox_dec'], syntax=syntax)
        self.pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        self.vox_enc = get_module_class(net_config['vox_enc']['model'], False)(net_config['vox_enc'], syntax=syntax)
        self.res_enc = get_module_class(net_config['res_enc']['model'], False)(net_config['res_enc'], syntax=syntax)

    def forward(self, coords):
        # Construct coordnates from sparse tensor
        coords[0][0] = 0
        coords[:, 0] = torch.cumsum(coords[:,0], 0)
        device = coords.device
        x = ME.SparseTensor(
            features=torch.ones(coords.shape[0], 1, device=device, dtype=torch.float32),
            coordinates=coords, 
            device=device)
        
        # This is to emulate the base layer
        with torch.no_grad():
            # Quantization
            x_coarse = scale_sparse_tensor_batch(x, factor=self.scaling_ratio)
            x_coarse = sort_sparse_tensor_with_dir(x_coarse)
            # x_coarse is supposed to be encoded losslessly here, followed by dequantization
            x_coarse_deq = torch.hstack((x_coarse.C[:, 0:1], (x_coarse.C[:, 1:] / self.scaling_ratio)))
        
        # Downsample feature
        feat = self.res_enc(x.C, x_coarse_deq)
        x_feat = ME.SparseTensor(
            features=feat,
            coordinate_manager=x_coarse.coordinate_manager,
            coordinate_map_key=x_coarse.coordinate_map_key
        )
        y=self.vox_enc(x_feat)
        y_q, likelihood = get_likelihood(self.entropy_bottleneck, y)

        # Upsample feature
        feat = self.vox_dec(y_q, x_coarse)
        
        # Diffusion decoder
        res = self.res_dec.sample(feat.F)

    def get_loss(self, coords):
        # coords: [N, 4], [:,0] indicates batch index, the later 3 dims are 3d position
        
        #construct sparse tensor for MinkowskiEngine
        coords[0][0]=0
        coords[:,0] = torch.cumsum(coords[:,0], 0)
        device=coords.device
        x = ME.SparseTensor(
            features=torch.ones(coords.shape[0], 1, device=device, dtype=torch.float32),
            coordinates=coords, 
            device=device)
        
        # downsample the original pc to a coarse one by scaling ratio
        with torch.no_grad():
            # Quantization
            x_coarse = scale_sparse_tensor_batch(x, factor=self.scaling_ratio)
            x_coarse = sort_sparse_tensor_with_dir(x_coarse)
            # x_coarse is supposed to be encoded losslessly here, followed by dequantization
            x_coarse_deq = torch.hstack((x_coarse.C[:, 0:1], (x_coarse.C[:, 1:] / self.scaling_ratio)))

        # compute geometry residule for each point in the coarse set
        # the residue of each coarse point is encoded into a feature vector
        feat, res = self.res_enc(x.C, x_coarse_deq, True)
        # construct sparse tensor with feature vector attached
        x_feat = ME.SparseTensor(
            features=feat,
            coordinate_manager=x_coarse.coordinate_manager,
            coordinate_map_key=x_coarse.coordinate_map_key
        )
        # the sparse feature grid is furtherly downsampley by sparse convolution
        y=self.vox_enc(x_feat)
        # estimate the distribution of the encoded features, used for entropy encoding
        y_q, likelihood = get_likelihood(self.entropy_bottleneck, y)

        # recover the sparse feature grid from quantized-downsampled version
        feat = self.vox_dec(y_q, x_coarse)

        # for training, we don't perform the whole diffusion reconstruction process,
        # instead we apply the method from https://arxiv.org/abs/2006.11239
        diffusion_loss = self.res_dec.get_loss(res, feat.F)
        pass

    pass

def get_likelihood(entropy_bottleneck, data):
    data_F, likelihood = entropy_bottleneck(data.F, quantize_mode="noise")
    data_Q = ME.SparseTensor(
        features=data_F,
        coordinate_map_key=data.coordinate_map_key,
        coordinate_manager=data.coordinate_manager,
        device=data.device
    )
    return data_Q, likelihood