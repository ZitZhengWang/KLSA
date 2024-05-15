import argparse
import os

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # ============================== dataset准备 =========================================
    dataset = datasets.make(config['dataset'], **config['dataset_args'])

    save_path = os.path.dirname(config["load"].rstrip('/'))
    utils.set_log_path(save_path)
    utils.log("\n============================================= Start test ================================================")
    utils.log('dataset: {} (x{}), {}'.format(
            dataset[0][0].shape, len(dataset), dataset.n_classes))
    yaml.dump(config, open(os.path.join(save_path, 'config_test.yaml'), 'w'))

    if not args.sauc:    # sauc默认为False
        n_way = 5
    else:
        n_way = 2

    n_shot, n_query = config["shot"], 15
    n_batch = 400
    ep_per_batch = 2
    batch_sampler = CategoriesSampler(dataset.label, n_batch, n_way, n_shot + n_query, ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=8, pin_memory=True)

    # ======================================== model ===============================================
    if config.get('load') is None:
        model = models.make('meta-baseline', encoder=None)
    else:
        model = models.load(torch.load(config['load']))

        if config.get("test_k") is not None:
            model.method.neighbor_k = config['test_k']
            utils.log('test_k: {}'.format(config['test_k']))

        if config.get("test_threshold") is not None:
            model.threshold = config['test_threshold']
            utils.log('test_threshold: {}'.format(config['test_threshold']))


    if config.get('load_encoder') is not None:
        encoder = models.load(torch.load(config['load_encoder'])).encoder
        model.encoder = encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    model.eval()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # ===================================== testing ===================================================
    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}

    test_epochs = args.test_epochs    # 默认为 10
    np.random.seed(0)

    for epoch in range(1, test_epochs + 1):
        va_lst = []
        for global_imgs, local_imgs, _ in tqdm(loader, leave=False):

            # 处理全局块
            g_shot, g_query = fs.split_shot_query(global_imgs.cuda(), n_way, n_shot, n_query, ep_per_batch=ep_per_batch)
            global_img_shape = g_query.shape[-3:]
            global_batch_shape = g_query.shape[:-3]
            with torch.no_grad():
                g_query = model.encoder(g_query.view(-1, *global_img_shape))
            g_query.view(*global_batch_shape, -1)

            # 处理局部快
            l_shot, l_query = fs.split_shot_query(local_imgs.cuda(), n_way, n_shot, n_query, ep_per_batch=ep_per_batch)

            with torch.no_grad():
                if not args.sauc:

                    logits = model(l_shot, l_query, g_query).view(-1, n_way)
                    label = fs.make_nk_label(n_way, n_query, ep_per_batch=ep_per_batch).cuda()
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)

                    aves['vl'].add(loss.item(), len(local_imgs))
                    aves['va'].add(acc, len(local_imgs))
                    va_lst.append(acc)
                else:
                    x_shot = x_shot[:, 0, :, :, :, :].contiguous()
                    shot_shape = x_shot.shape[:-3]
                    img_shape = x_shot.shape[-3:]
                    bs = shot_shape[0]
                    p = model.encoder(x_shot.view(-1, *img_shape)).reshape(
                            *shot_shape, -1).mean(dim=1, keepdim=True)
                    q = model.encoder(l_query.view(-1, *img_shape)).view(
                            bs, -1, p.shape[-1])
                    p = F.normalize(p, dim=-1)
                    q = F.normalize(q, dim=-1)
                    s = torch.bmm(q, p.transpose(2, 1)).view(bs, -1).cpu()
                    for i in range(bs):
                        k = s.shape[1] // 2
                        y_true = [1] * k + [0] * k
                        acc = roc_auc_score(y_true, s[i])
                        aves['va'].add(acc, len(local_imgs))
                        va_lst.append(acc)

        utils.log('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f} (@{})'.format(
                epoch, aves['va'].item() * 100,
                mean_confidence_interval(va_lst) * 100,
                aves['vl'].item(), _[-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/test_few_shot_CUB5shot_localpatch.yaml')
    # parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--test-epochs', type=int, default=10)
    parser.add_argument('--sauc', action='store_true')
    parser.add_argument('--gpu', default='0')

    parser.add_argument('--loadpath', default="save/KLSANet/max-va.pth",
                        help="指定测试模型")

    parser.add_argument('--patch_size', type=int, default=26, help="test influence of patch_size")
    parser.add_argument('--patchNum', type=int, default=36, help="test influence of patch num")

    parser.add_argument('--test_k', type=int, default=None, help="test influence of k")
    parser.add_argument('--test_threshold', type=float, default=None, help="test influence of")
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    if args.loadpath!=None:
        config["load"] = args.loadpath

    if args.test_k != None:
        config["test_k"] = args.test_k

    if args.test_threshold != None:
        config["test_threshold"] = args.test_threshold


    # patch_size消融实验，如果args.patch_size!=None,则将config中的全部替换成对应尺寸
    if args.patch_size!=None:
        config["dataset_args"]["patch_size"] = args.patch_size

    if args.patchNum!=None:
        config["dataset_args"]["patch_num"] = args.patchNum

    utils.set_gpu(args.gpu)
    main(config)

