import torch
import time
import os
import sys

import torch
import torch.distributed as dist
from sklearn.metrics import mean_absolute_error

from utils import AverageMeter, calculate_accuracy, calculate_precision_and_recall, get_tp


def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None,
                distributed=False):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    maes = AverageMeter()

    targets_all = None
    outputs_all = None

    epoch_time = time.time()
    end_time = time.time()
    for i, (inputs, labels) in enumerate(data_loader):

        data_time.update(time.time() - end_time)

        # for old action detection dataloader
        # video_ids, segments, targets = labels

        # for rate pred and new action det. Skip_factor = None for action det.
        targets, rates, video_ids, segments = labels

        rates = rates.to(device, non_blocking=True)
        outputs = model(inputs)

        # track all targets/outputs
        if targets_all is None:
            targets_all = rates
            outputs_all = outputs
        else:
            targets_all = torch.cat((targets_all, rates), axis=0)
            outputs_all = torch.cat((outputs_all, outputs), axis=0)

        loss = criterion(outputs.squeeze(), rates)

        # import pdb; pdb.set_trace()

        # mae
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mae = mean_absolute_error(rates.detach().cpu().numpy(), outputs.detach().cpu().numpy().squeeze())
        maes.update(mae, inputs.size(0))

        losses.update(loss.item(), inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_logger is not None:

            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'mae': maes.val,
                'lr': current_lr
            })

        print('Epoch: [{0}][{1}/{2}], '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
              'Data {data_time.val:.3f} ({data_time.avg:.3f}), '
              'Loss {loss.val:.4f} ({loss.avg:.4f}), '
              'MAE {mae.val:.4f} ({mae.avg:.4f}), '.format(
                                                    epoch,
                                                    i + 1,
                                                    len(data_loader),
                                                    batch_time=batch_time,
                                                    data_time=data_time,
                                                    loss=losses,
                                                    mae=maes))

    epoch_time = time.time() - epoch_time
    print('Epoch: [{0}], '
          'Time {epoch_time:.1f} ({epoch_time:.1f}), '.format(
                        epoch,
                        epoch_time=epoch_time))

    # Careful! not updated for new metrics
    if distributed:
        print('distributed not supported yet...!')
        raise Exception

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'lr': current_lr,
            'mae': maes.avg
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/lr', current_lr, epoch)
        tb_writer.add_scalar('train/mae', maes.avg, epoch)
