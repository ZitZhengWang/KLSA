import torch
import torch.nn as nn
from .models import register
import torch.nn.functional as F
@register('KLSM')
class KLSM(nn.Module):
    """
    KLSM
    """
    def __init__(self, threshold, weight=False):
        super(KLSM, self).__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, g_query_fea, l_query):
        # l_query: torch.Size([3, 75, 36, 512])    query_globle_fea: [3, 75, 512]

        batch, n_wq, n_patches, dim = l_query.shape

        # 首先将 g_query_fea 和 l_query做归一化处理，并将g_query_fea在最后一个维度扩展
        g_query_fea = F.normalize(g_query_fea, dim=-1).unsqueeze(-1)    # [3, 75, 512, 1]
        l_query = F.normalize(l_query, dim=-1)    # [3, 75, 36, 512]

        # 将g_query_fea 和 l_query 的前两个维度合并，以便进行 bmm操作，bmm要求三维张亮
        g_query_fea = g_query_fea.view(batch*n_wq, dim, -1)
        l_query = l_query.view(batch*n_wq, n_patches, dim)

        # 计算每张图像的局部块与 全局块 做cos相似度计算
        innerproduct_matrix = torch.bmm(l_query, g_query_fea)  # [3*75, 36, 512] * [3*75, 512, 1] = [3*75, 36, 1]

        # 根据相似度 和 阈值来得到 mask
        mask = (innerproduct_matrix > self.threshold) + 0.0  # 3维张量

        if self.weight:
            weight = innerproduct_matrix * mask    # [225,36,1]
            weight = F.normalize(weight, p=1, dim=-2)    # 1范数归一化的权重
            l_query = l_query * weight    # [225, 36, 512] * [225,36,1]
            l_query = F.normalize(l_query, dim=-1)    # 加权以后重新归一化
        else:
            l_query = l_query * mask  # [3*75, 36, 512] * [3*75, 36, 1] =

        l_query = l_query.view(batch, n_wq, n_patches, dim)    # [3, 75, 36, 512]
        return l_query
