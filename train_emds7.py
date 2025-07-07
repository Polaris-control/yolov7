#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 微藻检测训练脚本
专门针对 EMDS7 数据集优化
"""

import argparse
import logging
import os
import sys
import yaml
import time
import random
import math
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from threading import Thread

# 添加 yolov7 目录到路径
sys.path.append('yolov7')

import torch
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# PyTorch 2.6+ 兼容性：允许反序列化 numpy 对象和模型类
import numpy as np
import torch.serialization
torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])

# 添加 YOLOv7 模型类到安全全局变量
try:
    from yolov7.models.yolo import Model
    torch.serialization.add_safe_globals(['yolov7.models.yolo.Model'])
except ImportError:
    pass
from yolov7.models.yolo import Model
from yolov7.utils.datasets import create_dataloader
from yolov7.utils.general import (labels_to_class_weights, increment_path, 
                                 init_seeds, fitness, strip_optimizer, 
                                 check_dataset, check_file, set_logging, 
                                 colorstr, one_cycle, labels_to_image_weights)
#from yolov7.utils.google_utils import attempt_download
from yolov7.utils.loss import ComputeLoss, ComputeLossOTA
from yolov7.utils.plots import plot_images, plot_labels, plot_results


from yolov7.utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, attempt_load, de_parallel
#from yolov7.utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from yolov7.test import test


logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):
    """训练函数"""
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    
    # 初始化变量
    rank = -1  # 单GPU训练
    t0 = time.time()
    final_epoch = False
    wandb_logger = None
    loggers = {'wandb': None}
    
    # 保存目录
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # 保存运行设置
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # 配置
    plots = not opt.evolve
    cuda = device.type != 'cpu'
    init_seeds(2)
    
    # 加载数据集配置
    with open(opt.data) as f:
        data_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    # 从配置文件中提取数据集信息
    if isinstance(data_config, dict):
        data_keys = ['train', 'val', 'test', 'nc', 'names']
        data_dict = {}
        for key in data_keys:
            if key in data_config:
                data_dict[key] = data_config[key]
            else:
                # 使用默认值
                if key == 'train':
                    data_dict[key] = './EMDS7_min/images/train'
                elif key == 'val':
                    data_dict[key] = './EMDS7_min/images/val'
                elif key == 'test':
                    data_dict[key] = './EMDS7_min/images/test'
                elif key == 'nc':
                    data_dict[key] = 6
                elif key == 'names':
                    data_dict[key] = ['Oscillatoria', 'Microcystis', 'sphaerocystis', 
                                     'tribonema', 'brachionus', 'Ankistrodesmus']
    else:
        data_dict = {
            'train': './EMDS7_min/images/train',
            'val': './EMDS7_min/images/val',
            'test': './EMDS7_min/images/test',
            'nc': 6,
            'names': ['Oscillatoria', 'Microcystis', 'sphaerocystis', 
                     'tribonema', 'brachionus', 'Ankistrodesmus']
        }
    
    logger.info(f"数据集配置: {data_dict}")

    # 类别信息
    nc = int(data_dict['nc'])
    names = data_dict['names']
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset'

    # 模型
    pretrained = opt.weights.endswith('.pt')
    if pretrained:
        # 已有本地权重，不再 attempt_download
        pass
        #with torch_distributed_zero_first(-1):
            #attempt_download(opt.weights)
        # 使用 weights_only=False 来兼容旧版本权重文件
        try:
            ckpt = torch.load(opt.weights, map_location=device, weights_only=False)
            model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'), hyp=hyp).to(device)
            exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []
            state_dict = ckpt['model'].float().state_dict()
            state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f'从 {opt.weights} 加载了 {len(state_dict)}/{len(model.state_dict())} 个参数')
            if len(state_dict) < len(model.state_dict()) * 0.5:
                logger.warning(f"加载的参数数量远小于模型参数总数，可能权重与模型结构不匹配。建议检查模型结构或使用 --weights '' 进行随机初始化训练。")
        except Exception as e:
            logger.warning(f"使用 weights_only=False 加载失败: {e}")
            logger.warning(f"将使用随机初始化模型进行训练。建议检查权重文件与模型结构是否匹配。")
            model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), hyp=hyp).to(device)
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), hyp=hyp).to(device)

    # 检查数据集
    with torch_distributed_zero_first(-1):
        check_dataset(data_dict)
    
    # 验证路径
    for split in ['train', 'val', 'test']:
        if split in data_dict:
            if not os.path.exists(data_dict[split]):
                raise FileNotFoundError(f"数据集分割 '{split}' 路径错误: {data_dict[split]}")

    train_path = data_dict['train']
    test_path = data_dict['val']

    # 优化器
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / opt.batch_size), 1)
    hyp['weight_decay'] *= opt.batch_size * accumulate / nbs
    logger.info(f"缩放后的 weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, torch.nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        optimizer = torch.optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = torch.optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    logger.info(f'优化器组: {len(pg2)} .bias, {len(pg1)} conv.weight, {len(pg0)} other')
    del pg0, pg1, pg2

    # 学习率调度器
    if opt.linear_lr:
        lf = lambda x: (1 - x / (opt.epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], opt.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model)

    # 恢复训练
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
        start_epoch = ckpt['epoch'] + 1
        del ckpt

    # 图像尺寸
    gs = max(int(model.stride.max()), 32)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]

    # 数据加载器
    dataloader, dataset = create_dataloader(train_path, imgsz, opt.batch_size, gs, opt,
                                          hyp=hyp, augment=True, cache=opt.cache_images, 
                                          rect=opt.rect, rank=-1, world_size=1, 
                                          workers=opt.workers, image_weights=opt.image_weights, 
                                          quad=opt.quad, prefix=colorstr('train: '))

    # 验证数据加载器
    testloader = create_dataloader(test_path, imgsz_test, opt.batch_size * 2, gs, opt,
                                 hyp=hyp, cache=opt.cache_images and not opt.notest, 
                                 rect=True, rank=-1, world_size=1, workers=opt.workers,
                                 pad=0.5, prefix=colorstr('val: '))[0]

    # 损失函数
    if hyp.get('loss_ota', 1):
        compute_loss = ComputeLossOTA(model)
    else:
        compute_loss = ComputeLoss(model)

    # 开始训练
    logger.info(f'开始训练，共 {opt.epochs} 轮')
    logger.info(f'图像尺寸: {imgsz}')
    logger.info(f'批次大小: {opt.batch_size}')
    logger.info(f'累积步数: {accumulate}')
    
    # 训练循环
    nb = len(dataloader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5:.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model) if hyp.get('loss_ota', 1) else None
    compute_loss = ComputeLoss(model) if compute_loss_ota is None else None
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test')
    logger.info(f'Using {dataloader.num_workers} dataloader workers')
    logger.info(f'Logging results to {save_dir}')
    logger.info(f'Starting training for {opt.epochs} epochs...')
    
    # 训练循环
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        
        # Update image weights (optional)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 + maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
        
        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders
        
        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / opt.batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
            
            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                if compute_loss_ota is not None:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size
                if opt.quad:
                    loss *= 4.
            
            # Backward
            scaler.scale(loss).backward()
            
            # Optimize (with gradient accumulation)
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
            
            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    f'{epoch}/{opt.epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)
                
                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        
        # DDP mode
        if rank != -1:
            if rank == 0:
                torch.save(model, wdir / 'last.pt')
            dist.barrier()
        
        # mAP
        if not opt.notest or epoch == opt.epochs - 1:  # Calculate mAP
            results, maps, times = test(opt.data,
                                       batch_size=opt.batch_size * 2,
                                       imgsz=imgsz_test,
                                       conf_thres=0.001,
                                       iou_thres=0.65,  # for NMS
                                       save_json=False,
                                       model=ema.ema if ema else model,
                                       single_cls=opt.single_cls,
                                       dataloader=testloader,
                                       save_dir=save_dir,
                                       save_conf=False,
                                       plots=False,
                                       compute_loss=compute_loss)
        
        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
        
        # Log
        tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                'x/lr0', 'x/lr1', 'x/lr2']  # params
        for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard
            # if wandb_logger.wandb:
            #     wandb_logger.log({tag: x})  # W&B
        
        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5:.95]
        if fi > best_fitness:
            best_fitness = fi
        
        # Save model
        if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
            ckpt = {'epoch': epoch,
                   'best_fitness': best_fitness,
                   'model': deepcopy(model).half(),
                   'ema': deepcopy(ema.ema).half(),
                   'updates': ema.updates,
                   'optimizer': optimizer.state_dict(),
                   # 'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                   'date': datetime.now().isoformat()}
            
            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            if opt.save_period > 0 and epoch % opt.save_period == 0:
                torch.save(ckpt, wdir / f'epoch{epoch}.pt')
            del ckpt
        
        # Stop early
        if opt.patience and epoch > opt.patience:
            if fi < best_fitness * 0.1:
                logger.info(f'Early stopping at epoch {epoch}')
                break
    
    if rank in [-1, 0]:
        logger.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    logger.info(f'\nValidating {f}...')
                    results, _, _ = test(opt.data,
                                         batch_size=opt.batch_size * 2,
                                         imgsz=imgsz_test,
                                         conf_thres=0.001,
                                         iou_thres=0.65,
                                         model=attempt_load(f, device).half(),
                                         single_cls=opt.single_cls,
                                         dataloader=testloader,
                                         save_dir=save_dir,
                                         save_conf=False,
                                         plots=False,
                                         compute_loss=compute_loss)
        
        # Plot results
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            # if wandb_logger.wandb:
            #     files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
            #     wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
            #                                 if (save_dir / f).exists()]})
        
        # Save model
        if opt.save_period == -1 and not opt.evolve:
            logger.info(f'Saving {f}...')
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                       'best_fitness': best_fitness,
                       'model': deepcopy(model).half(),
                       'ema': deepcopy(ema.ema).half(),
                       'updates': ema.updates,
                       'optimizer': optimizer.state_dict(),
                       # 'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                       'date': datetime.now().isoformat()}
                torch.save(ckpt, best)
        
        # Log final results
        logger.info(f'Results saved to {save_dir}')
        logger.info(f'Best mAP: {best_fitness:.4f}')
    
    torch.cuda.empty_cache()
    return results


def check_img_size(img_size, s=32):
    """验证图像尺寸是否为步长的倍数"""
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        logger.warning(f'图像尺寸 {img_size} 必须是最大步长 {s} 的倍数，更新为 {new_size}')
    return new_size


def make_divisible(x, divisor):
    """使 x 能被 divisor 整除"""
    return int(x + divisor / 2) // divisor * divisor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv7 微藻检测训练脚本')
    parser.add_argument('--weights', type=str, default='yolov7/weights/yolov7.pt', help='初始权重路径')
    parser.add_argument('--cfg', type=str, default='yolov7/cfg/training/yolov7.yaml', help='模型配置文件路径')
    parser.add_argument('--data', type=str, default='yolov7/data/emds7_min.yaml', help='数据集配置文件路径')
    parser.add_argument('--hyp', type=str, default='yolov7/data/hyp.emds7.yaml', help='超参数配置文件路径')
    parser.add_argument('--patience', type=int, default=0, help='提前终止训练的容忍轮数（0为不提前终止）')
    parser.add_argument('--start_epoch', type=int, default=0, help='从哪个epoch开始训练')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='训练和测试图像尺寸')
    parser.add_argument('--rect', action='store_true', help='矩形训练')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='恢复最近的训练')
    parser.add_argument('--nosave', action='store_true', help='只保存最终检查点')
    parser.add_argument('--notest', action='store_true', help='只测试最终轮次')
    parser.add_argument('--noautoanchor', action='store_true', help='禁用自动锚框检查')
    parser.add_argument('--evolve', action='store_true', help='超参数进化')
    parser.add_argument('--cache-images', action='store_true', help='缓存图像以加速训练')
    parser.add_argument('--image-weights', action='store_true', help='使用加权图像选择进行训练')
    parser.add_argument('--device', default='', help='cuda设备，如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--multi-scale', action='store_true', help='变化图像尺寸 +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='将多类数据作为单类训练')
    parser.add_argument('--adam', action='store_true', help='使用 torch.optim.Adam() 优化器')
    parser.add_argument('--sync-bn', action='store_true', help='使用 SyncBatchNorm，仅在 DDP 模式下可用')
    parser.add_argument('--workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--project', default='runs/train', help='保存到 project/name')
    parser.add_argument('--name', default='emds7_exp', help='保存到 project/name')
    parser.add_argument('--exist-ok', action='store_true', help='允许现有 project/name，不递增')
    parser.add_argument('--quad', action='store_true', help='四重数据加载器')
    parser.add_argument('--linear-lr', action='store_true', help='线性学习率')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='标签平滑 epsilon')
    parser.add_argument('--save_period', type=int, default=-1, help='每 "save_period" 轮记录模型')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='冻结层：yolov7 backbone=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='在 AP 计算中假设最大召回率为 1.0')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    opt = parser.parse_args()

    # 设置日志
    set_logging()
    
    # 检查文件
    opt.data = check_file(opt.data) if opt.data else None
    opt.cfg = check_file(opt.cfg) if opt.cfg else None
    opt.hyp = check_file(opt.hyp) if opt.hyp else None
    
    # 验证必要文件
    if not opt.data:
        raise FileNotFoundError(f"数据集配置文件不存在: {opt.data}")
    if not opt.cfg and not opt.weights:
        raise ValueError("必须指定 --cfg 或 --weights 参数")
    if not opt.hyp:
        raise FileNotFoundError(f"超参数配置文件不存在: {opt.hyp}")
        
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)

    # 设备
    device = select_device(opt.device, batch_size=opt.batch_size)

    # 超参数
    with open(opt.hyp) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        
    # 从配置文件中提取超参数
    if isinstance(config, dict):
        # 直接使用配置文件中的超参数
        hyp = config
    else:
        # 如果配置文件不是字典格式，使用默认超参数
        hyp = {
            'lr0': 0.01, 'lrf': 0.1, 'momentum': 0.937, 'weight_decay': 0.0005,
            'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
            'box': 0.05, 'cls': 0.3, 'cls_pw': 1.0, 'obj': 0.7, 'obj_pw': 1.0,
            'iou_t': 0.20, 'anchor_t': 4.0, 'anchors': 3, 'fl_gamma': 0.0,
            'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0,
            'translate': 0.2, 'scale': 0.9, 'shear': 0.0, 'perspective': 0.0,
            'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.15,
            'copy_paste': 0.0, 'paste_in': 0.15, 'loss_ota': 1
        }
    
    logger.info(f"加载超参数配置: {opt.hyp}")
    logger.info(f"超参数: {hyp}")

    # 训练
    logger.info("=" * 60)
    logger.info("YOLOv7 微藻检测训练开始")
    logger.info("=" * 60)
    logger.info(f"训练配置: {opt}")
    logger.info(f"设备: {device}")
    logger.info(f"批次大小: {opt.batch_size}")
    logger.info(f"训练轮数: {opt.epochs}")
    logger.info(f"图像尺寸: {opt.img_size}")
    logger.info("=" * 60)
    
    if not opt.evolve:
        tb_writer = None
        tb_writer = SummaryWriter(opt.save_dir)
        train(hyp, opt, device, tb_writer) 