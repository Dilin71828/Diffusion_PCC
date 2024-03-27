import torch
import sys
import os

from pccai.optim.pcc_loss import PccLossBase

class DiffusionLoss(PccLossBase):

    def __init__(self, loss_args, syntax):
        super().__init__(loss_args, syntax)

    def loss(self, net_in, net_out):
        loss_out = {}

        if 'likelihoods' in net_out and len(net_out['likelihoods']) > 0:
            self.bpp_loss(loss_out, net_out['likelihoods'], net_out['gt'].shape[0])
        else:
            loss_out['bpp_loss'] = torch.zeros((1,)).cuda()
        
        loss_out['diffusion_loss'] = net_out['diffusion_loss']
        loss_out['loss'] = self.alpha*loss_out['diffusion_loss'] + self.beta*loss_out['bpp_loss']
        return loss_out