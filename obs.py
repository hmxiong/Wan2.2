import os
import sys
import math
import time

import torch

class OBS_Wan:
    def __init__(self, layer, args):
        self.layer = layer
        self.dev = self.layer.weight.device

        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        # self.arg = args
        # self.no_compensate = args.no_compensate
        self.sum_weight = 0
    
    def add_batch(self, inp, out, W_new):

        W_old = self.sum_weight

        W_total = W_old + W_new
        self.H *= W_old / W_total

        self.sum_weight = W_total
        
        norm_factor = math.sqrt(2 / self.sum_weight)
        inp = norm_factor * inp.float()
        self.H += inp.matmul(inp.t())

    def fast_prune(self, sparsity, prune_n=0, prune_m = 0, 
                   block_size=128, percdamp=.01):
        W = self.layer.weight.data.clone()
        #TODO for different layer type the Weight matrix may need to flattened

        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, block_size):
            i2 = min(i1 + block_size, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0


                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # if DEBUG:
            #     self.layer.weight.data[:, :i2] = W[:, :i2]
            #     self.layer.weight.data[:, i2:] = W[:, i2:]
            #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
            #     print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))

        # if isinstance(self.layer, transformers.Conv1D):
        #     W = W.t()

        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


    def free(self):
        self.H = None
        torch.cuda.empty_cache()
        