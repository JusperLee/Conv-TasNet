# -*- encoding: utf-8 -*-
'''
@Filename    :Loss.py
@Time        :2020/07/09 22:11:13
@Author      :Kai Li
@Version     :1.0
'''

import torch
from itertools import permutations

class Loss(object):
    def __init__(self):
        super(Loss, self).__init__()

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        t = torch.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def compute_loss(self, ests, refs):

        def sisnr_loss(permute):
            # for one permute
            return sum(
                [self.sisnr(ests[s], refs[t])
                 for s, t in enumerate(permute)]) / len(permute)

        # P x N
        N = ests[0].size(0)
        sisnr_mat = torch.stack(
            [sisnr_loss(p) for p in permutations(range(len(ests)))])
        max_perutt, _ = torch.max(sisnr_mat, dim=0)
        # si-snr
        return -torch.sum(max_perutt) / N


if __name__ == "__main__":
    ests = torch.randn(4,320)
    egs = torch.randn(4,320)
    loss = Loss()
    print(loss.compute_loss(ests, egs))