import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    # 如果没有指定保存的文件名，则根据数据集以及主干网络、分类器等信息生成一个文件名；
    svname = args.name
    if svname is None:
        svname = 'classifier_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr

    # 为文件名添加额外的标签、注释
    if args.tag is not None:
        svname += '_' + args.tag

    # 由指定路径 和 文件名生成保存的路径；
    save_path = os.path.join('./save', svname)

    # 检查保存路径是否已经存在，如果存在是否删除旧路径
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)    # 设置日志文件的保存路径
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))    # 设置tensorboard的保存路径

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))    # 保存实验中 ymal 文件的配置情况（记录实验参数）

    #### Dataset ####

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=0, pin_memory=True)
    # 日志记录 数据集中一个图像经过处理后的样本的形状、数据的个数 以及 总类别数
    utils.log('train dataset: {}  {}  (x{}), {}'.format(
            train_dataset[0][0].shape, train_dataset[0][1].shape, len(train_dataset),  # train_dataset[0]表示从Dataset的getitem中获取索引为0结果，结果是一个元组（self.transform(self.data[i]), self.label[i]）包含处理后的图像及其标签；
            train_dataset.n_classes))
    # 如果 config中配置了 可视化数据集的选项，则利用tensorboard进行可视化
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)

    # val 传统验证数据集准备
    if config.get('val_dataset'):
        eval_val = True
        val_dataset = datasets.make(config['val_dataset'],
                                    **config['val_dataset_args'])    # 验证集的数据使用默认的预处理操作，没有数据增强
        val_loader = DataLoader(val_dataset, config['batch_size'],
                                num_workers=0, pin_memory=True)
        utils.log('val dataset: {}  {}  (x{}), {}'.format(
                val_dataset[0][0].shape,  val_dataset[0][1].shape,len(val_dataset),
                val_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    else:
        eval_val = False

    # few-shot eval 小样本任务验证数据集准备
    if config.get('fs_dataset'):
        ef_epoch = config.get('eval_fs_epoch')   # 每隔ef_epoch 个 epoch做一次 few-shot验证
        if ef_epoch is None:
            ef_epoch = 5
        eval_fs = True

        fs_dataset = datasets.make(config['fs_dataset'],
                                   **config['fs_dataset_args'])
        utils.log('fs dataset: {}  {}  (x{}), {}'.format(
                fs_dataset[0][0].shape, fs_dataset[0][0].shape, len(fs_dataset),
                fs_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(fs_dataset, 'fs_dataset', writer)

        n_way = 5
        n_query = 15
        n_shots = [1, 5]    # 1-shot 和 5-shot 都要测试
        fs_loaders = []

        # 构建出 1-shot 和 5-shot 的dataloader
        for n_shot in n_shots:
            fs_sampler = CategoriesSampler(
                fs_dataset.label,    # 20个测试类别的所有图像，共计12000张，的标签；
                200,    # 应该表示采样的任务数量
                n_way,    # n_way
                n_shot + n_query,    # n_shot + n_query
                ep_per_batch=4)    # 每个batch包含多少个任务
            fs_loader = DataLoader(fs_dataset, batch_sampler=fs_sampler,
                                   num_workers=0, pin_memory=True)
            fs_loaders.append(fs_loader)
    else:
        eval_fs = False

    ########

    #### Model and Optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])    # 加载模型，需要指定模型的名字和相应的参数

    if eval_fs:
        fs_model = models.make('meta-baseline', encoder=None, method_args={"neighbor_k": 5}, threshold=0.2)
        # fs_model 模型使用 和 model一样的特征提取器，只是分类器是一个余弦分类器
        # fs_model.encoder = model.encoder

    # 多GPU的并行设置
    if config.get('_parallel'):
        model = nn.DataParallel(model)
        if eval_fs:
            fs_model = nn.DataParallel(fs_model)
    # 计算模型的参数量，并且记录在日志中
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])

    ########
    
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    if args.use_amp:
        scalar = GradScaler()  # 1.这是你需要添加的

    for epoch in range(1, max_epoch + 1 + 1):    # [1,2,3,...,101]
        # 如果是最后一个额外的 epoch
        if epoch == max_epoch + 1:
            if not config.get('epoch_ex'):
                break
            train_dataset.transform = train_dataset.default_transform
            train_loader = DataLoader(
                    train_dataset, config['batch_size'], shuffle=True,
                    num_workers=0, pin_memory=True)

        timer_epoch.s()

        # 设置几个关键字，分别表示 训练损失、训练准确率、验证损失、验证准确率  #
        aves_keys = ['tl', 'ta', 'vl', 'va']

        # 添加 1-shot 和5-shot 的准确率关键字
        if eval_fs:
            for n_shot in n_shots:
                aves_keys += ['fsa-' + str(n_shot)]

        # 根据刚才设置的关键字生成对应的Averager
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)    # tensorboard记录学习了的变化情况

        # 所有train_loader中的所有数据训练一遍
        for global_imgs, local_imgs, label in tqdm(train_loader, desc='train', leave=False):

            local_imgs, label = local_imgs.cuda(), label.cuda()

            optimizer.zero_grad()
            if args.use_amp:
                with torch.cuda.amp.autocast():  # 2.这是你需要添加的
                    logits = model(local_imgs)    # [128, 64]
                    loss = F.cross_entropy(logits, label)

                scalar.scale(loss).backward()  # 3.这是你需要添加的,进行损失缩放工作
                scalar.step(optimizer)  # 4.这是你需要添加的
                scalar.update()  # 5.这是你需要添加的

            else:
                logits = model(local_imgs)  # [128, 64]
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = utils.compute_acc(logits, label)

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            logits = None; loss = None

        # eval
        if eval_val:
            model.eval()
            # val_loader中的所有数据训练一遍
            for global_imgs, local_imgs, label in tqdm(val_loader, desc='val', leave=False):
                local_imgs, label = local_imgs.cuda(), label.cuda()    # 注意，一个batch的验证图像都来自同一个类
                with torch.no_grad():
                    logits = model(local_imgs)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                
                aves['vl'].add(loss.item())
                aves['va'].add(acc)

        #
        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            fs_model.eval()
            for i, n_shot in enumerate(n_shots):

                # 设置固定的种子，做few-shot验证
                np.random.seed(0)    # data [320, 3, 80, 80]
                for global_imgs, local_imgs, _ in tqdm(fs_loaders[i],
                                    desc='fs-' + str(n_shot), leave=False):
                    # 处理全局块
                    g_shot, g_query = fs.split_shot_query(global_imgs.cuda(), n_way, n_shot, n_query,
                                                          ep_per_batch=4)
                    global_img_shape = g_query.shape[-3:]
                    global_batch_shape = g_query.shape[:-3]
                    with torch.no_grad():
                        g_query = model.encoder(g_query.view(-1, *global_img_shape))
                    g_query.view(*global_batch_shape, -1)

                    # 处理局部快
                    l_shot, l_query = fs.split_shot_query(local_imgs.cuda(), n_way, n_shot, n_query,
                                                          ep_per_batch=4)

                    label = fs.make_nk_label(n_way, n_query, ep_per_batch=4).cuda()    # [300, ]
                    with torch.no_grad():
                        fs_model.encoder = model.encoder
                        logits = fs_model(l_shot, l_query, g_query).view(-1, n_way)    # [300, 5]
                        acc = utils.compute_acc(logits, label)   # 计算一个batch 4个任务中query 的准确率
                    aves['fsa-' + str(n_shot)].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()    # 返回数值结果，存在字典中

        t_epoch = utils.time_str(timer_epoch.t())    # 每个epoch的用时
        t_used = utils.time_str(timer_used.t())    # 使用的时间
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)    # 估计总用时

        if epoch <= max_epoch:
            epoch_str = str(epoch)
        else:
            epoch_str = 'ex'
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                epoch_str, aves['tl'], aves['ta'])    # 显示的内容依次为：epoch数，训练损失，训练准确率
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)

        if eval_val:
            log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])    # 显示内容分别为验证损失，验证准确率
            writer.add_scalars('loss', {'val': aves['vl']}, epoch)
            writer.add_scalars('acc', {'val': aves['va']}, epoch)

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            log_str += ', fs'
            for n_shot in n_shots:
                key = 'fsa-' + str(n_shot)
                log_str += ' {}: {:.4f}'.format(n_shot, aves[key])
                writer.add_scalars('acc', {key: aves[key]}, epoch)

        if epoch <= max_epoch:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)    # 追加显示 一个epoch训练用时，一轮用时，估计总用时
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model
        # 保存训练过程 优化器的状态
        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }    # 其他保存的事项
        save_obj = {
            'file': __file__,    # 保存运行的文件名
            'config': config,    # 保存参数设置

            'model': config['model'],    # 保存模型名称
            'model_args': config['model_args'],    # 保存模型的参数设置
            'model_sd': model_.state_dict(),    # 保存模型的参数状态

            'training': training,   # 以及优化器的参数状态
        }
        if epoch <= max_epoch:
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))

            if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(save_obj, os.path.join(
                    save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))

        writer.flush()


if __name__ == '__main__':
    # 通过argparse 和 yaml解析实验中的参数；
    # argparse主要包括需要经常变动的参数；
    # yaml中主要包括数据加载、模型相关的参数，不需要经常变动；
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train_classifier_folder_localpatch.yaml", help="配置文件的路径")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--use_amp', action="store_false", help="使用混合精度训练")
    args = parser.parse_args()

    # 将yaml中的参数加载为一个字典的形式，存放在config中；
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    # 配置多GPU的情况
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    # 设置GPU
    utils.set_gpu(args.gpu)

    main(config)

