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
main.py runs this, but this script was heavily modified to save results to disk for analyzing.

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


Note:  analyzing saved predictions vs labels is on Notebook on Eric's local computer, can ask him.

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

