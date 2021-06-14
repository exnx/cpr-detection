# Compression detection and rate prediction

This repo contains code for training 1. chest compression detection and 2. rate prediction. 


### 1. Compression Detection

##### Important subdirs:
* #### classifier/3D-ResNets-Pytorch
* #### post_processing_code/

```
classifier/3D-ResNets-Pytorch/create_segments.py
```
  
Create the sliding window segments and its corresponding labels.

```
classifier/3D-ResNets-Pytorch/datasets/cpr_clip_dataset.py:
```  
Uses the segment and labels to retrieve images for training.

```
classifier/3D-ResNets-Pytorch/main_detection.py
``` 
Runs the training for compression detection, and the inference (separate step).  

```
classifier/3D-ResNets-Pytorch/inference.py
```
main.py runs this, but this script was heavily modified to save results to disk.  This inference script already performs prec/recall (unlike the rate inference code).

##### post_processing_code/  

```
video_to_frame_and_meta.py
```
Converts videos to frames and gathers the metadata in a json.

```
write_action_on_frames.py
```
Reads predictions and labels and writes results on frames, and saves.

```
create_video_from_frames.py
```
Creates videos from frames.


### 2. Rate prediction

```
classifier/3D-ResNets-Pytorch/datasets/slow_varied_dataset.py:
```  
Dataset loads (time cropped) compression clips. This dataset uses 0.2x speed clips, but then skips frames by a factor (varied) depending on the label, slow or fast (or normal). 

Note: segments are created on the fly, no need to create beforehand.

```
classifier/3D-ResNets-Pytorch/main_rate_mse.py
```
Used to train rate prediction using MSE loss (instead of binary classification).

```
classifier/3D-ResNets-Pytorch/inference_mse.py
```

Uses MSE loss on inference for rate prediction (only), saves results.

##### post_processing_code/ 

```
post_process_inference_results_rate.py (for rate only)
```
This is used to "bucketize" the inference results into classification preds and labels. Then performs prec/recall/f1.

Possibly use run_metrics_on_inf.py from /old, should be same.

```
render_frames_rate.py
```
Writes the preds/labels onto frames.

```
create_video_from_frames.py
```
Same as above



### Data stored on SC node and Macondo3 (for SVL only)

##### /vision2/u/enguyen/results

Lots of results for compression detection.  Latest (best) is: 

`pretrain1_cont2`

The checkpoints, logs are in here.

Below tells you a lot:

python write_action_on_frames.py \
--results-dir /vision2/u/enguyen/results/pretrain1_cont2/val_1/val.json \
--frame-dir /vision2/u/enguyen/mini_cba/new_fps10/ \
--out-dir /vision2/u/enguyen/mini_cba/frames_with_text_no_alert2/

- Inference results on in val_1/val.json
- Frames used are at 10 FPS

##### /vision2/u/enguyen/results/rate_pred

`/vision2/u/enguyen/results/rate_pred/run8_res18_mse_action_pretrained`

This contains the best rate predictor.  It used the pretrained model from the compression detector to start.  Uses MSE loss also.

##### /vision2/u/enguyen/demos

This director contains frames with results written on them, but also directories with just the inference results (no frames written), for Res and RepNet.

`/vision2/u/enguyen/demos/rate_pred/run8_chpt24`

This directory has the best results, with frames, and videos created. Other run8 folders here were drafts basically.

##### Macondo3

This node has frames saved to disk for faster training, at various FPS, basically 16 fps for rate prediction, and full (30) fps for inference.

##### All videos location

`/vision/group/miniCBA/download_video/videos/allVideos`





