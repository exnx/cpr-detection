import torch
import time
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy, calculate_precision_and_recall, get_tp


def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              logger,
              tb_writer=None,
              distributed=False):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    targets_all = None
    outputs_all = None
    video_ids_all = None

    true_pos_count = 0
    union_count = 0

    epoch_time = time.time()
    end_time = time.time()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            # for old action detection dataloader
            # video_ids, segments, targets = labels

            # for rate pred and new action det. Skip_factor = None for action det.
            targets, rates, video_ids, segments = labels

            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)

            # add preds
            _, preds = torch.max(outputs.data, 1)

            batch_tp_count = get_tp(preds, targets)
            true_pos_count += batch_tp_count

            # sum of targets==1, and sum of preds==1
            union_count += (torch.sum(targets).item() + torch.sum(preds).item() - batch_tp_count)

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            # track all targets/outputs
            if targets_all is None:
                targets_all = targets
                outputs_all = outputs
                # video_ids_all = video_ids
            else:
                targets_all = torch.cat((targets_all, targets), axis=0)
                outputs_all = torch.cat((outputs_all, outputs), axis=0)
                # video_ids_all = torch.cat((video_ids_all, video_ids), axis=0)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}], '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}), '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                  'Acc {acc.val:.3f} ({acc.avg:.3f}), '.format(
                                                                epoch,
                                                                i + 1,
                                                                len(data_loader),
                                                                batch_time=batch_time,
                                                                data_time=data_time,
                                                                loss=losses,
                                                                acc=accuracies))

            # if i >= 75:
            #     break

    tiou = true_pos_count / union_count

    # calc stats
    precision, recall, f1 = calculate_precision_and_recall(outputs_all, targets_all)

    # save outputs, targets, ids to disk

    epoch_time = time.time() - epoch_time
    print('Epoch: [{0}], '
          'Time {epoch_time:.1f} ({epoch_time:.1f}), '
          'precision {precision:.2f}, '
          'recall {recall:.2f}, '
          'f1 {f1:.2f}, '
          'tIOU {tiou:.2f}, '.format(
                        epoch,
                        epoch_time=epoch_time,
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        tiou=tiou))

    # Careful! not updated for new metrics
    if distributed:
        print('distributed not supported yet...!')
        raise Exception

    if logger is not None:
        logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tiou': tiou
        })

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/acc', accuracies.avg, epoch)
        tb_writer.add_scalar('val/precision', precision, epoch)
        tb_writer.add_scalar('val/recall', recall, epoch)
        tb_writer.add_scalar('val/f1', f1, epoch)
        tb_writer.add_scalar('val/tiou', f1, epoch)

    return losses.avg
