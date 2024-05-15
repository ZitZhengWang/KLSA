import torch
import torch.nn as nn
from .models import register
import torch.nn.functional as F

@register('SSMM')
class SSMM(nn.Module):
    """
    SSMM
    """
    # l_shot: torch.Size([3, 5, 5, 36, 512])    l_query: torch.Size([3, 75, 36, 512])

    def __init__(self, neighbor_k=3):
        super(SSMM, self).__init__()
        self.neighbor_k = neighbor_k

    def cal_cosinesimilarity(self, query_fea, support_fea):
        # [3, 5, 180, 512]    [3, 75, 36, 512]
        # 【batch, way, shot*patch_num, dim】    【batch, n_wq, patch_num, dim】

        batch, n_wq, patch_num, dim = query_fea.shape
        _, n_way, _, dim = support_fea.shape

        Similarity_list = []

        for k in range(len(support_fea)):    # 遍历每一个batch
            for i in range(n_wq):  # 遍历每一个query

                query = query_fea[k][i]   # 【n_patches, dim】   第k个batch的 第 i个 query

                if torch.cuda.is_available():
                    inner_sim = torch.zeros(1, n_way).cuda()  # 创建一个空的向量，存放最终计算出的相似度

                for j in range(n_way):  # 遍历support中的每个类
                    support_class = support_fea[k][j]      # [n_support*n_patches, dim] 第k个batch的 第 j 个类

                    support_class = torch.transpose(support_class, 0, 1)     # [dim, n_support*n_patches]

                    innerproduct_matrix = query @ support_class  # 【n_patches, dim】* [dim, n_support*n_patches]

                    topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, dim=1)
                    inner_sim[0, j] = torch.sum(topk_value)    # topk_value、topk_index 【36, 3】 将全部的值加起来；

                Similarity_list.append(inner_sim)

        Similarity_list = torch.cat(Similarity_list, 0)

        return Similarity_list  # [75, 5]


    def forward(self, query_fea, support_fea):
        # query_fea: torch.Size([3, 75, 36, 512]) 归一化后的
        # support_fea    [3, 5, 5, 36, 512]  归一化后的

        batch, way, shot, patch_num, dim = support_fea.shape
        _, n_wq, _, _= query_fea.shape

        support_fea = support_fea.view(batch, way, shot*patch_num, dim)    # [3, 5, 180, 512]

        Similarity_list = self.cal_cosinesimilarity(query_fea, support_fea)

        return Similarity_list
