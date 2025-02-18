import math

import torch
import torch.nn as nn

import models
import utils
from .models import register


@register('classifier')
class Classifier(nn.Module):
    
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)    # 依然使用 make函数来加载特征提取器；把特征提取器视为一个子模型
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)    # 同样使用 make函数来加载分类器，把分类器视为一个子模型；

    def forward(self, x):
        x = self.encoder(x)    # 输出 x【128，512】
        x = self.classifier(x)    # 输出 x【128，64】
        return x


@register('linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


@register('nn-classifier')
class NNClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, metric='cos', temp=None):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temp is None:
            if metric == 'cos':
                temp = nn.Parameter(torch.tensor(10.))
            else:
                temp = 1.0
        self.metric = metric
        self.temp = temp

    def forward(self, x):
        return utils.compute_logits(x, self.proto, self.metric, self.temp)

