#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
from torch.nn import Softmax
import torch.nn as nn


class ConsistencyCos(nn.Module): 
    def __init__(self):
        super(ConsistencyCos, self).__init__()
        self.mse_fn = nn.MSELoss()
        self.l1_fn = nn.L1Loss()

    def forward(self, feat):
        feat = nn.functional.normalize(feat, dim=1)

        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:] 
        cos = torch.einsum('nc,nc->n', [feat_0, feat_1]).unsqueeze(-1)
        labels = torch.ones((cos.shape[0],1), dtype=torch.float, requires_grad=False)
        if torch.cuda.is_available(): 
            labels = labels.cuda()
        loss = self.mse_fn(cos, labels)
        #loss = self.l1_fn(cos, labels)
        return loss

def entropy(input_): 
    bs = input_.size(0)
    epsilon = 1e-10
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 


def im(outputs_test, gent=True): 
    epsilon = 1e-10
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    entropy_loss = torch.mean(entropy(softmax_out))
    if gent:
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log2(msoftmax + epsilon))
        entropy_loss -= gentropy_loss
    im_loss = entropy_loss * 1.0
    return im_loss


def adv(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(features.device)
    return torch.nn.BCELoss()(ad_out, dc_target)

def adv_local(features, ad_net, is_source=False, weights=None): 
    ad_out = ad_net(features).squeeze(3)
    batch_size = ad_out.size(0)
    num_heads = ad_out.size(1)
    seq_len = ad_out.size(2)

    if is_source:
        label = torch.from_numpy(np.array([[[1] * seq_len] * num_heads] * batch_size)).float().to(features.device)
    else:
        label = torch.from_numpy(np.array([[[0] * seq_len] * num_heads] * batch_size)).float().to(features.device)

    return ad_out, torch.nn.BCELoss()(ad_out, label)


if __name__ == '__main__':

    batch_size = 16
    num_classes = 2
    logits = torch.randn(batch_size, num_classes, requires_grad=True)

    logitt = torch.randn(batch_size, num_classes, requires_grad=True)
    logt=torch.cat((logits,logits),dim=0)
    # print(logt)

    target = torch.randint(0, 2, (batch_size,))
    loss_im=im(logits)
