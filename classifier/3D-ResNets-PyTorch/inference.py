import time
import json
from collections import defaultdict

import copy
import torch
import torch.nn.functional as F

from utils import AverageMeter,  calculate_precision_and_recall, get_tp


def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = torch.topk(outputs,
                                     k=min(output_topk, len(class_names)))

    # import pdb; pdb.set_trace()

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results

def inference(data_loader, model, result_path, class_names, no_average,
              output_topk):
    print('inference')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    results = {'results': defaultdict(list)}

    targets_all = None
    outputs_all = None
    preds_all = None
    video_ids_all = None

    true_pos_count = 0
    union_count = 0

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):

            data_time.update(time.time() - end_time)

            # targets, video_ids, segments, skip_factors = labels
            targets, rates, video_ids, segments = labels
            
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1).cpu()

            # add preds
            _, preds = torch.max(outputs.data, 1)

            # true_pos_count += torch.eq(targets, preds).sum().item()
            batch_tp_count = get_tp(preds, targets)
            true_pos_count += batch_tp_count

            # sum of targets==1, and sum of preds==1
            union_count += (torch.sum(targets).item() + torch.sum(preds).item() - batch_tp_count)

            # import pdb; pdb.set_trace()

            for j in range(outputs.size(0)):
                results['results'][video_ids[j]].append({
                    'target': targets[j],
                    'pred': preds[j],
                    'segment': [segments[0][j].item(), segments[1][j].item()],
                    'output': [outputs[j][0].item(), outputs[j][1].item()]
                })

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            # track all targets/outputs
            if targets_all is None:
                targets_all = targets
                outputs_all = outputs
                preds_all = preds
                video_ids_all = video_ids
            else:
                targets_all = torch.cat((targets_all, targets), axis=0)
                outputs_all = torch.cat((outputs_all, outputs), axis=0)
                preds_all = torch.cat((preds_all, preds), axis=0)
                video_ids_all = video_ids_all + video_ids

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))

            # if i >= 100:
            #     break

    # import pdb; pdb.set_trace()

    # need to write json of all the outputs...!

    tiou = true_pos_count / union_count

    # calc stats
    precision, recall, f1 = calculate_precision_and_recall(outputs_all, targets_all)

    print('precision {precision:.2f}, '
          'recall {recall:.2f}, '
          'f1 {f1:.2f}, ' 
          'tIOU {tiou:.2f}, '.format(
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        tiou=tiou))

    inference_results = {'results': {}, 'precision': precision, 'recall': recall, 'f1': f1, 'tiou': tiou}

    # this isn't working anymore
    if not no_average:
        for video_id, video_results in results['results'].items():
            video_outputs = [
                segment_result['output'] for segment_result in video_results
            ]
            video_outputs = torch.stack(video_outputs)
            average_scores = torch.mean(video_outputs, dim=0)
            inference_results['results'][video_id] = get_video_results(
                average_scores, class_names, output_topk)

    # default to this
    else:
        for video_id, video_results in results['results'].items():
            # inference_results['results'][video_id] = []
            # for segment_result in video_results:
            #     segment = segment_result['segment']
            #     target = segment_result['target']
            #     # result = get_video_results(segment_result['output'],
            #     #                            class_names, output_topk)
            #     pred = segment_result['pred']
            #     inference_results['results'][video_id].append({
            #         'target': target.item(),
            #         # 'result': result,
            #         'segment': segment,
            #         'pred': pred.item(),
            #     })
            
            video_dict = {'targets': [], 'preds': [], 'segments': [], 'outputs': []}
            
            for segment_result in video_results:
                segment = segment_result['segment']
                target = segment_result['target']
                output = segment_result['output']
                # result = get_video_results(segment_result['output'],
                #                            class_names, output_topk)
                pred = segment_result['pred']
                
                video_dict['targets'].append(target.item())
                video_dict['segments'].append(segment)
                video_dict['preds'].append(pred.item())
                video_dict['outputs'].append(output)
                
                # inference_results['results'][video_id].append({
                #     'target': target.item(),
                #     # 'result': result,
                #     'segment': segment,
                #     'pred': pred.item(),
                # })
                #

            # probably at video level stats
            # need to save outputs too to get pr/rc, f1, tIOU

            inference_results['results'][video_id] = copy.deepcopy(video_dict)


    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=4)
