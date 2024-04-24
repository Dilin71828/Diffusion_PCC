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
from third_party.PCGCv2.gpcc import gpcc_encode, gpcc_decode
from third_party.PCGCv2.data_utils import read_ply_ascii_geo, write_ply_ascii_geo
from third_party.PCGCv2.data_utils import scale_sparse_tensor

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
        net_config['res_dec']['feature_dim'] = net_config['vox_dec']['dims'][-1]

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
        feat, geo_res = self.res_enc(x.C, x_coarse_deq, output_res=True)
        x_feat = ME.SparseTensor(
            features=feat,
            coordinate_manager=x_coarse.coordinate_manager,
            coordinate_map_key=x_coarse.coordinate_map_key
        )
        y=self.vox_enc(x_feat)
        y_q, likelihood = get_likelihood(self.entropy_bottleneck, y)

        # Upsample feature
        feat = self.vox_dec(y_q, x_coarse)

        diffusion_loss = self.res_dec.get_loss(geo_res*2/self.thres_dist, feat.F)  # need to normalize the geo residue into [-2,2]
        
        # Diffusion decoder
        return {'gt': coords,
                'likelihoods': {'feats': likelihood},
                'diffusion_loss': diffusion_loss}
    
    def compress(self, x, tag):
        """
        This function performs actual compression with learned statistics of the entropy bottleneck, consumes one point cloud at a time.
        """

        # Start the compression here
        x_coarse = scale_sparse_tensor(x, factor=self.scaling_ratio)
        filename_base = tag + '_B.bin'
        print("Start base encoding...")
        start = time.monotonic()
        coord_codec(filename_base, x_coarse.C.detach().cpu()[:, 1:]) # encode with G-PCC losslessly
        base_enc_time = time.monotonic() - start
        print(f"base encode completed in {base_enc_time}s")

        x_coarse_deq = (x_coarse.C[:, 1:] / self.scaling_ratio).float().unsqueeze(0).contiguous()
        x_c = x.C[:, 1:].float().unsqueeze(0)
        del x

        torch.cuda.empty_cache()
        print("Start encoding residue...")
        feat = self.res_enc(x_c, x_coarse_deq)
        x_feat = ME.SparseTensor( # low bitdepth PC with attr
                    features=feat,
                    coordinate_manager=x_coarse.coordinate_manager,
                    coordinate_map_key=x_coarse.coordinate_map_key)
        
        print("Start voxel encoding...")
        y = self.vox_enc(x_feat) # voxel encoder
        y = sort_sparse_tensor_with_dir(y)
        shape = y.F.shape
        string, min_v, max_v = self.entropy_bottleneck.compress(y.F.cpu())

        return filename_base, [string], [min_v], [max_v], [shape], x_coarse.shape[0], base_enc_time
    
    def decompress(self, filename_base, string, min_v, max_v, shape, base_dec_time):
        """
        This function performs actual decompression with learned statistics of the entropy bottleneck, consumes one point cloud at a time.
        """
        print("Decode base stream...")
        start = time.monotonic()
        y_C = coord_codec(filename_base) # decode with G-PCC losslessly
        base_dec_time[0] = time.monotonic() - start
        y_C = torch.cat((torch.zeros((len(y_C), 1)).int(), torch.tensor(y_C).int()), dim=-1)

        # From y_C, create the downsampled versions of y_C for decoding
        device = next(self.parameters()).device
        y_dummy = ME.SparseTensor(
                features=torch.ones((y_C.shape[0], 1), device=device),
                coordinates=y_C, 
                tensor_stride=1, 
                device=device
            )
        del y_C
        torch.cuda.empty_cache()

        y_ds = self.pool(y_dummy if self.dus == 1 else self.pool(y_dummy))
        y_ds = sort_sparse_tensor_with_dir(y_ds)
        y_F = self.entropy_bottleneck.decompress(string[0], min_v[0], max_v[0], shape[0], channels=shape[0][-1])
        y_down = ME.SparseTensor(features=y_F, device=device,
                coordinate_manager=y_ds.coordinate_manager,
                coordinate_map_key=y_ds.coordinate_map_key)
        
        print("Decode features...")
        y_dec = self.vox_dec(y_down, y_dummy) # feature decoder
        y_dec_C = (y_dec.C[:, 1:] / self.scaling_ratio).float().contiguous()

        print("Diffusion decoding...")
        with torch.no_grad():
            decoded_res = self.res_dec.sample(y_dec.F, x_coarse = y_dec_C*2/self.thres_dist,
                                           start_step = self.res_dec.start_step)
        print(f"Debug: decoded res: max{decoded_res.max()}, min{decoded_res.min()}")
        out = y_dec_C.repeat_interleave(self.point_mul, dim=0) + decoded_res.reshape(-1,3)*self.thres_dist/2
        return out

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
        return {'diffusion_loss': diffusion_loss,
                'likelihood': likelihood,
                'gt': coords}
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

def coord_codec(bin_filename, coords=None):
    ply_filename = bin_filename + '.ply'
    if coords == None: # decode
        gpcc_decode(bin_filename, ply_filename)
        out = read_ply_ascii_geo(ply_filename)
    else: # encode
        coords = coords.numpy().astype('int')
        write_ply_ascii_geo(filedir=ply_filename, coords=coords)
        gpcc_encode(ply_filename, bin_filename)
        out = bin_filename
    os.system('rm '+ ply_filename)
    return out