#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from xmuda.common.solver.build import build_optimizer, build_scheduler
from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build_BEVDG import build_model_2d, build_model_3d
from xmuda.data.build import build_dataloader
from xmuda.data.utils.validate import validate
from xmuda.data.utils.bev_fusion import BEV_fusion
from xmuda.models.losses import con_loss
from torch import nn

def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meters(new_metric_list)
    return metric_logger

# DVM
def generate_vector(cat_feats, unq_cnt):
    one = torch.ones_like(unq_cnt)
    zero = torch.zeros_like(unq_cnt)
    mask1 = torch.where(unq_cnt<=10, one, zero)
    mask3 = torch.where(unq_cnt>50, one, zero)
    mask2 = one-mask1-mask3
    ratio1 = mask1.sum().item() / unq_cnt.shape[0]
    ratio2 = mask2.sum().item() / unq_cnt.shape[0]
    ratio3 = mask3.sum().item() / unq_cnt.shape[0]
    vector1 = (cat_feats * mask1.view(-1, 1).cuda()).max(dim=0)
    vector2 = (cat_feats * mask2.view(-1, 1).cuda()).max(dim=0)
    vector3 = (cat_feats * mask3.view(-1, 1).cuda()).max(dim=0)
    vector = vector1[0] * ratio1 + vector2[0] * ratio2 + vector3[0] * ratio3
    return vector

def train(cfg, output_dir='', run_name=''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    logger = logging.getLogger('xmuda.train')

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model_2d, train_metric_2d = build_model_2d(cfg)
    logger.info('Build 2D model:\n{}'.format(str(model_2d)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    # build 3d model
    model_3d, train_metric_3d = build_model_3d(cfg)
    logger.info('Build 3D model:\n{}'.format(str(model_3d)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build optimizer
    optimizer_2d = build_optimizer(cfg, model_2d)
    optimizer_3d = build_optimizer(cfg, model_3d)

    # build lr scheduler
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer_2d = CheckpointerV2(model_2d,
                                     optimizer=optimizer_2d,
                                     scheduler=scheduler_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_2d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_2d = checkpointer_2d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_3d = CheckpointerV2(model_3d,
                                     optimizer=optimizer_3d,
                                     scheduler=scheduler_3d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_3d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_3d = checkpointer_3d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data_2d.get('iteration', 0)

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration)
    train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='target') if val_period > 0 else None

    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        '2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None)
    }
    best_metric_iter = {'2d': -1, '3d': -1}
    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d])
    val_metric_logger = MetricLogger(delimiter='  ')

    def setup_train():
        # set training mode
        model_2d.train()
        model_3d.train()
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model_2d.eval()
        model_3d.eval()
        # reset metric
        val_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    setup_train()
    end = time.time()
    train_iter_src = enumerate(train_dataloader_src)
    train_iter_trg = enumerate(train_dataloader_trg)
    for iteration in range(start_iteration, max_iteration):
        # fetch data_batches for source & target
        _, data_batch_src = train_iter_src.__next__()
        _, data_batch_trg = train_iter_trg.__next__()
        data_time = time.time() - end
        # copy data from cpu to gpu
        if 'SCN' in cfg.DATASET_SOURCE.TYPE and 'SCN' in cfg.DATASET_TARGET.TYPE:
            # source
            data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
            data_batch_src['seg_label'] = data_batch_src['seg_label'].cuda()
            data_batch_src['img'] = data_batch_src['img'].cuda()
            # target
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
            data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch_trg['img'] = data_batch_trg['img'].cuda()
            if cfg.TRAIN.XMUDA.lambda_pl > 0:
                data_batch_trg['pseudo_label_2d'] = data_batch_trg['pseudo_label_2d'].cuda()
                data_batch_trg['pseudo_label_3d'] = data_batch_trg['pseudo_label_3d'].cuda()
        else:
            raise NotImplementedError('Only SCN is supported for now.')

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()

        # ---------------------------------------------------------------------------- #
        # Density Transfer
        # ---------------------------------------------------------------------------- #
        img_s = data_batch_src['img']
        img_t = data_batch_trg['img']
        pc_st = []
        pc_ts = []
        last_number_s = 0
        last_number_t = 0
        for i in range(img_s.shape[0]):
            n_src = data_batch_src['img_indices'][i].shape[0]  # 第i个源域点云里点的数量
            current_number_s = last_number_s + n_src
            xyz_s = data_batch_src['x'][0][last_number_s:current_number_s, :3]  # 第i个源域点云里点的坐标
            x_s = torch.max(xyz_s[:, 0]) - torch.min(xyz_s[:, 0])
            y_s = torch.max(xyz_s[:, 1]) - torch.min(xyz_s[:, 1])
            z_s = torch.max(xyz_s[:, 2]) - torch.min(xyz_s[:, 2])
            density_s = (x_s * y_s * z_s) / n_src

            n_trg = data_batch_trg['img_indices'][i].shape[0]  # 第i个目标域点云里点的数量
            current_number_t = last_number_t + n_trg
            xyz_t = data_batch_trg['x'][0][last_number_t:current_number_t, :3]  # 第i个目标点云里点的坐标
            x_t = torch.max(xyz_t[:, 0]) - torch.min(xyz_t[:, 0])
            y_t = torch.max(xyz_t[:, 1]) - torch.min(xyz_t[:, 1])
            z_t = torch.max(xyz_t[:, 2]) - torch.min(xyz_t[:, 2])
            density_t = (x_t * y_t * z_t) / n_trg

            alfa = (density_s.item() / density_t.item()) ** (-(1 / 3))
            xyz_st = xyz_s * alfa
            xyz_st = torch.LongTensor(xyz_st.numpy())

            beta = (density_t.item() / density_s.item()) ** (-(1 / 3))
            xyz_ts = xyz_t * beta
            xyz_ts = torch.LongTensor(xyz_ts.numpy())

            index_s = data_batch_src['x'][0][last_number_s:current_number_s, 3].view(-1, 1)
            pc_st.append(torch.cat([xyz_st, index_s], dim=1))

            index_t = data_batch_trg['x'][0][last_number_t:current_number_t, 3].view(-1, 1)
            pc_ts.append(torch.cat([xyz_ts, index_t], dim=1))

            last_number_s += n_src
            last_number_t += n_trg
        pc_st = torch.cat(pc_st, 0)
        pc_st = [pc_st, data_batch_src['x'][1]]
        data_batch_st = {
            'x': pc_st,
            'seg_label': data_batch_src['seg_label'],
            'img': img_s,
            'img_indices': data_batch_src['img_indices']
        }

        pc_ts = torch.cat(pc_ts, 0)
        pc_ts = [pc_ts, data_batch_trg['x'][1]]
        data_batch_ts = {
            'x': pc_ts,
            'seg_label': data_batch_trg['seg_label'],
            'img': img_t,
            'img_indices': data_batch_trg['img_indices']
        }
        # ---------------------------------------------------------------------------- #
        # Train on source1
        # ---------------------------------------------------------------------------- #

        feats_2d = model_2d(data_batch_src)
        feats_3d = model_3d(data_batch_src)
        feats_3d_st = model_3d(data_batch_st)

        last_number = 0
        batch_bev_feats = []
        batch_vector = []
        batch_vector_st = []
        for i in range(data_batch_src['img'].shape[0]):
            indices_pt = data_batch_src['img_indices'][i]
            num_pt = indices_pt.shape[0]
            cur_number = last_number + num_pt
            xyz_pt = data_batch_src['x'][0][last_number:cur_number, :3]
            xyz_pt_st = data_batch_st['x'][0][last_number:cur_number, :3]
            feats_2d_pt = feats_2d[last_number:cur_number, :]
            feats_3d_pt = feats_3d[last_number:cur_number, :]
            feats_3d_pt_st = feats_3d_st[last_number:cur_number, :]
            # BEV fusion
            cat_feats, pool_3d, unq_inv, unq_cnt = BEV_fusion(xyz_pt, feats_2d_pt,feats_3d_pt)
            cat_feats_st, pool_3d_st, unq_inv_st, unq_cnt_st = BEV_fusion(xyz_pt_st, feats_2d_pt, feats_3d_pt_st)
            unq_inv = unq_inv.unsqueeze(1).repeat(1, cat_feats.shape[1])
            point_feats = cat_feats.gather(dim=0, index=unq_inv)
            batch_bev_feats.append(point_feats)
            # DVM
            voxe_feats = generate_vector(cat_feats, unq_cnt)
            voxe_feats_st = generate_vector(cat_feats_st, unq_cnt_st)
            batch_vector.append(voxe_feats.view(1,-1))
            batch_vector_st.append(voxe_feats_st.view(1, -1))
            last_number = cur_number
        batch_bev_feats = torch.cat(batch_bev_feats, dim=0)
        batch_vector = torch.cat(batch_vector,dim=0)
        batch_vector_st = torch.cat(batch_vector_st, dim=0)

        preds_2d = model_2d(data_batch_src, feats_2d, batch_bev_feats)
        preds_3d = model_3d(data_batch_src, feats_3d, batch_bev_feats)

        vector_src1 = F.normalize(batch_vector,dim=1)
        vector_st = F.normalize(batch_vector_st, dim=1)

        labels_src1 = torch.tensor([0,1,2,3,4,5,6,7])
        labels_st = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        # segmentation loss: cross entropy
        seg_loss_src1_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        seg_loss_src1_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        train_metric_logger.update(seg_loss_src1_2d=seg_loss_src1_2d, seg_loss_src1_3d=seg_loss_src1_3d)
        loss_2d_src1 = seg_loss_src1_2d
        loss_3d_src1 = seg_loss_src1_3d


        if cfg.TRAIN.XMUDA.lambda_xm_src > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_src_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_src_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_src_2d=xm_loss_src_2d,
                                       xm_loss_src_3d=xm_loss_src_3d)
            loss_2d_src1 += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_2d
            loss_3d_src1 += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_3d

        # update metric (e.g. IoU)
        with torch.no_grad():
            train_metric_2d.update_dict(preds_2d, data_batch_src)
            train_metric_3d.update_dict(preds_3d, data_batch_src)

        loss_src1 = loss_2d_src1 + loss_3d_src1
        # ---------------------------------------------------------------------------- #
        # Train on source2
        # ---------------------------------------------------------------------------- #

        feats_2d = model_2d(data_batch_trg)
        feats_3d = model_3d(data_batch_trg)
        feats_3d_ts = model_3d(data_batch_ts)

        last_number = 0
        batch_bev_feats = []
        batch_vector = []
        batch_vector_ts = []
        for i in range(data_batch_trg['img'].shape[0]):
            indices_pt = data_batch_trg['img_indices'][i]
            num_pt = indices_pt.shape[0]
            cur_number = last_number + num_pt
            xyz_pt = data_batch_trg['x'][0][last_number:cur_number, :3]
            xyz_pt_ts = data_batch_ts['x'][0][last_number:cur_number, :3]
            feats_2d_pt = feats_2d[last_number:cur_number, :]
            feats_3d_pt = feats_3d[last_number:cur_number, :]
            feats_3d_pt_ts = feats_3d_ts[last_number:cur_number, :]
            # BEV fusion
            cat_feats, unq_inv, unq_cnt = BEV_fusion(xyz_pt, feats_2d_pt, feats_3d_pt)
            cat_feats_ts, unq_inv_ts, unq_cnt_ts = BEV_fusion(xyz_pt_ts, feats_2d_pt, feats_3d_pt_ts)
            unq_inv = unq_inv.unsqueeze(1).repeat(1, cat_feats.shape[1])
            point_feats = cat_feats.gather(dim=0, index=unq_inv)
            batch_bev_feats.append(point_feats)
            # DVM
            voxe_feats = generate_vector(cat_feats, unq_cnt)
            batch_vector.append(voxe_feats.view(1, -1))
            voxe_feats_ts = generate_vector(cat_feats_ts, unq_cnt_ts)
            batch_vector_ts.append(voxe_feats_ts.view(1, -1))
            last_number = cur_number
        batch_bev_feats = torch.cat(batch_bev_feats, dim=0)  # [48911,80]
        batch_vector = torch.cat(batch_vector, dim=0)
        batch_vector_ts = torch.cat(batch_vector_ts, dim=0)

        vector_src2 = F.normalize(batch_vector, dim=1)
        vector_ts = F.normalize(batch_vector_ts, dim=1)
        labels_src2 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        labels_ts = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        preds_2d = model_2d(data_batch_trg, feats_2d, batch_bev_feats)
        preds_3d = model_3d(data_batch_trg, feats_3d, batch_bev_feats)

        loss_2d_src2 = []
        loss_3d_src2 = []
        # segmentation loss: cross entropy
        seg_loss_src2_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch_trg['seg_label'], weight=class_weights)
        seg_loss_src2_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_trg['seg_label'], weight=class_weights)
        train_metric_logger.update(seg_loss_src2_2d=seg_loss_src2_2d, seg_loss_src2_3d=seg_loss_src2_3d)
        loss_2d_src2.append(seg_loss_src2_2d)
        loss_3d_src2.append(seg_loss_src2_3d)
        if cfg.TRAIN.XMUDA.lambda_xm_trg > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_trg_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_trg_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                       xm_loss_trg_3d=xm_loss_trg_3d)
            loss_2d_src2.append(cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_trg_2d)
            loss_3d_src2.append(cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_trg_3d) # xmloss的权重要和源域一样

        loss_src2 = sum(loss_2d_src2) + sum(loss_3d_src2)
        # contrastive loss
        vectors_s1 = torch.cat([vector_src1, vector_st], dim=0)
        labels_s1 = torch.cat([labels_src1, labels_st], dim=0)
        contrast_loss_s1 = 0.01*con_loss(vectors_s1, labels_s1)
        train_metric_logger.update(contrast_loss_s1=contrast_loss_s1)

        vectors_s2 = torch.cat([vector_src2, vector_ts], dim=0)
        labels_s2 = torch.cat([labels_src2, labels_ts], dim=0)
        contrast_loss_s2 = 0.01 * con_loss(vectors_s2, labels_s2)
        train_metric_logger.update(contrast_loss_s2=contrast_loss_s2)

        contrast_loss = contrast_loss_s1 + contrast_loss_s2

        total_loss = loss_src1 + loss_src2 + contrast_loss
        total_loss.backward()

        optimizer_2d.step()
        optimizer_3d.step()

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)

        # log
        cur_iter = iteration + 1
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr=optimizer_2d.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
            checkpoint_data_2d['iteration'] = cur_iter
            checkpoint_data_2d[best_metric_name] = best_metric['2d']
            checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **checkpoint_data_2d)
            checkpoint_data_3d['iteration'] = cur_iter
            checkpoint_data_3d[best_metric_name] = best_metric['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **checkpoint_data_3d)

        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if val_period > 0 and cur_iter>=80000 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            start_time_val = time.time()
            setup_validate()

            validate(cfg,
                     model_2d,
                     model_3d,
                     val_dataloader,
                     val_metric_logger)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in ['2d', '3d']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric[modality] is None or best_metric[modality] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter

            # restore training
            setup_train()

        scheduler_2d.step()
        scheduler_3d.step()
        end = time.time()

    for modality in ['2d', '3d']:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(modality.upper(),
                                                                     cfg.VAL.METRIC,
                                                                     best_metric[modality] * 100,
                                                                     best_metric_iter[modality]))


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('xmuda', output_dir, comment='train.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # check that 2D and 3D model use either both single head or both dual head
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    assert cfg.TRAIN.XMUDA.lambda_xm_src > 0 or cfg.TRAIN.XMUDA.lambda_xm_trg > 0 or cfg.TRAIN.XMUDA.lambda_pl > 0 or \
           cfg.TRAIN.XMUDA.lambda_minent > 0
    train(cfg, output_dir, run_name)


if __name__ == '__main__':
    main()