
import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler


def main(config):
    # ================================处理实验记录保存问题==================================
    svname = args.name
    if svname is None:
        svname = 'meta_{}-{}shot'.format(
                config['train_dataset'], config['n_shot'])
        svname += '_' + config['model'] + '-' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)    # svname仅仅为了组成保存的路径
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    # ================================Dataset准备工作=======================================
    # 设置小样本参数
    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    # 训练的way 和 shot 可能和测试不一样，所以设置了 n_train_way、n_train_shot 这两个参数
    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1
    patch_num = config['train_dataset_args']["patch_num"]
    # train dataset
    # 因为在我们自己的实验中需要使用局部块 和 全局块，所以数据集部分要有一个自己的版本
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    utils.log('train dataset: {}  {}  (x{}), {}'.format(
            train_dataset[0][0].shape, train_dataset[0][1].shape, len(train_dataset),
            train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)

    # CategoriesSampler中设置了一个batch的图像如何去构建，包含任务的构建过程
    train_sampler = CategoriesSampler(
        train_dataset.label,    # 所用数据部分的全部图像的标签
        config['train_batches'],    # 训练的batch数量，使用到的部分的数据全部运行 200次；
        n_train_way,
        n_train_shot + n_query,
        ep_per_batch=ep_per_batch)    # 每个batch 包含的 任务数量
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    # tval    真正测试数据集的构建
    if config.get('tval_dataset'):
        tval_dataset = datasets.make(config['tval_dataset'],
                                     **config['tval_dataset_args'])
        utils.log('tval dataset: {}  {}  (x{}), {}'.format(
                tval_dataset[0][0].shape, tval_dataset[0][1].shape, len(tval_dataset),
                tval_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
        tval_sampler = CategoriesSampler(
            tval_dataset.label,
            200,
            n_way,
            n_shot + n_query,
            ep_per_batch=1)
        tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                 num_workers=8, pin_memory=True)
    else:
        tval_loader = None

    # val    验证用的数据集
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {}  {}  (x{}), {}'.format(
            val_dataset[0][0].shape, val_dataset[0][1].shape, len(val_dataset),
            val_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(
        val_dataset.label,
        200,
        n_way,
        n_shot + n_query,
        ep_per_batch=1)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    # ================================ Model and optimizer =======================================

    if config.get('load'):    # load 是加载整个模型，包括特征提取器和分类器
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):    # load_encoder 加载特征提取器部分
            encoder = models.load(torch.load(config['load_encoder'])).encoder    # config['load_encoder']是一个路径，torch.load(config['load_encoder'])是一个保存的字典；models.load(torch.load(config['load_encoder']))构建一个待加载的模型，并且加载其中的参数，并且，把模型中encoder的参数提取出来，用于当前的模型；
            model.encoder.load_state_dict(encoder.state_dict())    # 给模型的encoder部分加载参数；

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])

    # =================================== 训练部分================================================
    
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    if args.use_amp:
        scalar = GradScaler()  # 1.这是你需要添加的

    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}    # 在每一轮训练开始以后再定义 计算loss 和准确率的计算器

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model) 
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        np.random.seed(epoch)    # 随着epoch的变化，改变随机种子；
        for global_imgs, local_imgs, _ in tqdm(train_loader, desc='train', leave=False):    # 所有数据训练一轮
            # global_imgs [B,C,H,W]    images [B, patch_num, C, H, W]

            # 从Dataloader中得到数据以后，把global_imgs 和 local_imgs的 支持集和查询集分开；不需要修改split_shot_query函数
            # g_query：torch.Size([3, 75, 3, 84, 84])
            # 我们只需要全局图像的query部分，并提取他们的特征；
            g_shot, g_query = fs.split_shot_query(
                    global_imgs.cuda(), n_train_way, n_train_shot, n_query,
                    ep_per_batch=ep_per_batch)
            global_img_shape = g_query.shape[-3:]
            global_batch_shape = g_query.shape[:-3]
            with torch.no_grad():
                g_query = model.encoder(g_query.view(-1, *global_img_shape))
            g_query.view(*global_batch_shape, -1)

            # l_shot: torch.Size([3, 5, 5, 36, 3, 21, 21])    l_query: torch.Size([3, 75, 36, 3, 21, 21])
            l_shot, l_query = fs.split_shot_query(
                    local_imgs.cuda(), n_train_way, n_train_shot, n_query,
                    ep_per_batch=ep_per_batch)

            label = fs.make_nk_label(n_train_way, n_query,
                    ep_per_batch=ep_per_batch).cuda()

            optimizer.zero_grad()
            if args.use_amp:
                with autocast():  # 2.这是你需要添加的
                    logits = model(l_shot, l_query, g_query).view(-1, n_train_way)
                    loss = F.cross_entropy(logits, label)

                scalar.scale(loss).backward()  # 3.这是你需要添加的,进行损失缩放工作
                scalar.step(optimizer)  # 4.这是你需要添加的
                scalar.update()  # 5.这是你需要添加的
            else:
                logits = model(l_shot, l_query, g_query).view(-1, n_train_way)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = utils.compute_acc(logits, label)

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            logits = None; loss = None 

        # ============================ eval ========================================
        model.eval()
        # 遍历 测试 和 验证数据集
        for name, loader, name_l, name_a in [
                ('tval', tval_loader, 'tvl', 'tva'),
                ('val', val_loader, 'vl', 'va')]:

            if (config.get('tval_dataset') is None) and name == 'tval':
                continue

            np.random.seed(0)
            for global_imgs, local_imgs, _ in tqdm(loader, desc=name, leave=False):

                g_shot, g_query = fs.split_shot_query(
                    global_imgs.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=1)
                global_img_shape = g_query.shape[-3:]
                global_batch_shape = g_query.shape[:-3]
                with torch.no_grad():
                    g_query = model.encoder(g_query.view(-1, *global_img_shape))
                g_query.view(*global_batch_shape, -1)

                # l_shot: torch.Size([3, 5, 5, 36, 3, 21, 21])    l_query: torch.Size([3, 75, 36, 3, 21, 21])
                l_shot, l_query = fs.split_shot_query(
                    local_imgs.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=1)

                label = fs.make_nk_label(n_way, n_query,
                        ep_per_batch=1).cuda()

                with torch.no_grad():
                    logits = model(l_shot, l_query, g_query).view(-1, n_way)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                
                aves[name_l].add(loss.item())
                aves[name_a].add(acc)

    # ===================================== 结果记录=====================================
        _sig = int(_[-1])    # 最后一个 val 图像的真实标签 给 _sig

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
                'val {:.4f}|{:.4f}, {} {}/{} (@{})'.format(
                epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
                aves['vl'], aves['va'], t_epoch, t_used, t_estimate, _sig))

        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train_meta_CUB_5shot_localpatch.yaml", help="配置文件的路径")
    parser.add_argument('--name', default="KLSANet", help="保存模型的名字")
    parser.add_argument('--tag', default=None, help="对模型名字的补充")
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--use_amp', action="store_false", help="使用混合精度训练")
    parser.add_argument('--weight', action="store_true", help="消融实验使用")
    parser.add_argument('--use_BGF', action="store_false", help="KLSM")

    parser.add_argument('--patch_size', type=int, default=26, help="test influence of patch_size")
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--neighbor_k', type=int, default=5, help="分类部分KNN的k值的大小")
    parser.add_argument('--patchNum', type=int, default=1, help="test influence of patch num")
    parser.add_argument("--train_augment", type=str, default="Crop_Resize")
    parser.add_argument('--encoder', type=str, default="resnet12", help="opt conv")

    parser.add_argument('--load_encoder', type=str, default="save/classifier_image_folder_spec_resnet12/max-va.pth", help="指定预训练的特征提取器路径")
    # parser.add_argument('--load_encoder', type=str, default=None, help="指定预训练的特征提取器路径")

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    config['use_amp'] = args.use_amp
    config["load_encoder"] = args.load_encoder

    config["model_args"]["use_BGF"] = args.use_BGF
    config["model_args"]["threshold"] = args.threshold
    config["model_args"]["weight"] = args.weight
    config["model_args"]["encoder"] = args.encoder

    config["model_args"]["method_args"]["neighbor_k"] = args.neighbor_k
    # patch_size消融实验，如果args.patch_size!=None,则将config中的全部替换成对应尺寸
    if args.patch_size!=None:
        config["train_dataset_args"]["patch_size"] = args.patch_size
        config["tval_dataset_args"]["patch_size"] = args.patch_size
        config["val_dataset_args"]["patch_size"] = args.patch_size

    if args.patchNum!=None:
        config["train_dataset_args"]["patch_num"] = args.patchNum
        config["tval_dataset_args"]["patch_num"] = args.patchNum
        config["val_dataset_args"]["patch_num"] = args.patchNum

    config["train_dataset_args"]["augment"] = args.train_augment

    utils.set_gpu(args.gpu)
    main(config)

