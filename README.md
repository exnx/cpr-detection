# cpr-detection

##### Important subdirs:  
* #### dataset/  
* #### post_processing_code/


##### dataset/:

```
create_segments.py
```
  
Create the sliding window segments and its corresponding labels.

```
video_dataset.py:
```  
Uses the segment and labels to retrieve images for training.



##### post_processing_code/  

```
video_to_frame_and_meta.py
```
Converts videos to frames and gathers the metadata in a json.
