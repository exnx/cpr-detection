import time
import json
from collections import defaultdict

import copy
import torch
import torch.nn.functional as F

from utils import AverageMeter, calculate_precision_and_recall, get_tp


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

    outputs_all = None
    video_ids_all = None

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):

            data_time.update(time.time() - end_time)

            # targets, video_ids, segments, skip_factors = labels
            targets, rates, video_ids, segments = labels

            outputs = model(inputs)

            for j in range(outputs.size(0)):
                results['results'][video_ids[j]].append({
                    'segment': [segments[j][0].item(), segments[j][1].item()],
                    'output': [outputs[j].item()]
                })

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            # track all targets/outputs
            if outputs_all is None:
                outputs_all = outputs
                video_ids_all = video_ids
            else:
                outputs_all = torch.cat((outputs_all, outputs), axis=0)
                video_ids_all = video_ids_all + video_ids

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time))

    # need to write json of all the outputs...!
    inference_results = {'results': {}}

    # default to this

    for video_id, video_results in results['results'].items():
        video_dict = {'preds': [], 'segments': [], 'outputs': []}

        for segment_result in video_results:
            segment = segment_result['segment']
            output = segment_result['output']

            video_dict['segments'].append(segment)
            video_dict['outputs'].append(output)

        inference_results['results'][video_id] = copy.deepcopy(video_dict)

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=4)
