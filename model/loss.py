import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import Encoder, generate_dboxes


class SigmoidDRLoss(nn.Module):
    def __init__(self, pos_lambda=1, neg_lambda=0.1/math.log(3.5), L=6., tau=4.):
        super(SigmoidDRLoss, self).__init__()
        self.margin = 0.5
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.L = L
        self.tau = tau

    def forward(self, logits, targets, classes):
        t = torch.zeros(4, 4, 8732)
        tar = targets.unsqueeze(1)

        # pad ground truth with 0
        for i in range(len(classes)):
            t[i][classes[i]] = tar[i]

        # num_classes = logits.shape[1]
        # dtype = targets.dtype
        # device = targets.device
        # class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)
        class_range = [0, 1, 2, 3]

        pos_ind = (t == class_range)
        # pos_ind = t > 0
        neg_ind = (t != class_range) * (t >= 0)
        pos_prob = logits[pos_ind].sigmoid()
        neg_prob = logits[neg_ind].sigmoid()
        # print(pos_prob)
        # print(neg_prob)
        neg_q = F.softmax(neg_prob/self.neg_lambda, dim=0)
        neg_dist = torch.sum(neg_q * neg_prob)
        if pos_prob.numel() > 0:
            pos_q = F.softmax(-pos_prob/self.pos_lambda, dim=0)
            pos_dist = torch.sum(pos_q * pos_prob)
            loss = self.tau*torch.log(1.+torch.exp(self.L*(neg_dist - pos_dist+self.margin)))/self.L
        else:
            loss = self.tau*torch.log(1.+torch.exp(self.L*(neg_dist - 1. + self.margin)))/self.L
        return loss


def compute_iou(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def prepare_data(ploc, gloc, plabel):
    encoder = Encoder(generate_dboxes())
    p = ploc.detach().cpu().float()
    pl = plabel.detach().cpu().float()
    g = gloc.detach().cpu().float()

    p, _ = encoder.scale_back_batch(p, pl)
    g, _ = encoder.scale_back_batch(g, pl)

    return p, g


def find_weights(ploc, gloc, lam=1., w=0.01):
    """finds iou weights for input type Nx8732x4 of ltrb bbox format"""
    p = ploc.detach().cpu().numpy()
    g = gloc.detach().cpu().numpy()

    weights = []

    for j in range(ploc.shape[0]):  # batch
        clweights = []
        gbboxes = np.array(g[j])
        for i in range(ploc.shape[1]):
            pbbox = np.array([p[j][i]])
            if pbbox[0][0] < pbbox[0][2] and pbbox[0][1] < pbbox[0][3]:
                clweights.append(w * np.power(compute_iou(pbbox, gbboxes).sum(), lam))
            else:
                clweights.append(0.0)
        weights.append(clweights)
    return torch.tensor(weights)


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, dboxes, use_weighted_iou=True):
        super(Loss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh
        self.use_weighted_iou = use_weighted_iou

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0), requires_grad=False)
        self.con_loss = nn.CrossEntropyLoss(reduction='none')

    def loc_vec(self, loc):
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, ]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels
            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        torch.autograd.set_detect_anomaly(True)
        mask = glabel > 0
        pos_num = mask.sum(dim=1)
        vec_gd = self.loc_vec(gloc)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)
        # print(con)

        if self.use_weighted_iou:
            pl, gl = prepare_data(ploc, gloc, plabel)
            weights = find_weights(pl, gl)
            w = sl1/(sl1.clone() + weights.cuda())
            sl1 = w * sl1

        sl1 = (mask.float() * sl1).sum(dim=1)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        closs = (con * (mask.float() + neg_mask.float())).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)

        return ret
