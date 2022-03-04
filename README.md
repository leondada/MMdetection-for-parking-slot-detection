### MMdetection codes for DMPR-PS 停车位检测(parking-slot-detection)”

This is the mmdetection-implementation of DMPR-PS based on [DMPR-PS](https://github.com/Teoge/DMPR-PS)

### Step-1 Install mmcv 1.4.5 and MMdetection 2.19.0

### Step-2 Download the dataset 

Prepare the dataset and preprocess it according to DMPR-PS 

Aggregate annotation files into one json, like this:
```
import os
import mmcv
import json
anns = list()
root = '/mnt/disk2/FedPSDet/Server/label/test/'
for file in os.listdir(root):
    if file.endswith(".json"):
        sample = {}
        sample['filename'] = os.path.splitext(file)[0]+'.jpg'
        sample['width'] = 600
        sample['height'] = 600
        sample['ann'] = mmcv.load(root+file)
        anns.append(sample)
with open('./test_psd.json', 'w') as file:
    json.dump(anns, file)
```

### Step-3 Change the code 

Overwrite our codes into the mmdetection file. 

### Step-4 Change the configs/FedPSD/yolo.py
 
### Step-5 Train

```
bash ./tools/dist_train.sh
```
8张1080ti，每张卡12个样本，迭代1000次后即有0.906的ap（计算方式与DMPR完全一致）
