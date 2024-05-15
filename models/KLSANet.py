import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .KLSM import KLSM
from .models import register


@register('KLSANet')
class KLSANet(nn.Module):
    """
    KLSANet
    """
    def __init__(self, encoder, encoder_args={}, method='SSMM', method_args={}, use_BGF=True, threshold=0, weight=False,
                 temp=1., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.use_BGF = use_BGF

        if self.use_BGF:
            self.KLSM = KLSM(threshold=threshold, weight=weight)

        if method=="SSMM":
            self.method = models.make(method, **method_args)
        else:
            self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query, g_query_fea=None):
        # l_shot: torch.Size([3, 5, 5, 36, 3, 21, 21])
        # l_query: torch.Size([3, 75, 36, 3, 21, 21])
        shot_shape = x_shot.shape[:-3]    # [1, 5, 5, 36]
        query_shape = x_query.shape[:-3]    # [1, 75, 36]
        img_shape = x_shot.shape[-3:]    # [3,21,21]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))    # [320, 512]
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)    # l_shot: torch.Size([3, 5, 5, 36, 512])
        x_query = x_query.view(*query_shape, -1)    # l_query: torch.Size([3, 75, 36, 512])

        if self.use_BGF:
            x_query = self.KLSM(g_query_fea, x_query)
        else:
            x_query = F.normalize(x_query, dim=-1)  # 加权以后重新归一化

        x_shot = F.normalize(x_shot, dim=-1)    # [3, 5, 5, 36, 512]

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)

            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
            logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'
            logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        else:
            logits = self.method(x_query, x_shot) * self.temp


        return logits

