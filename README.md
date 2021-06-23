# Multiple Object Tracking Based on The Simulation of Physical Locations

This repository hosts our code for our paper Multiple Object Tracking Based on The Simulation of Physical Locations. 

The code will be released after the paper published.

## Introduction

Tracking-by-detection is a widely used paradigm to solve the problem of multiple object tracking (MOT). This paper presents a tracking method that incorporates simulated physical locations of people into the Simple Online and Realtime Tracking with a Deep association metric (Deep SORT) algorithm. A learning-based framework is applied to extract the appearance features across frames in a video sequence. The proposed method predicts the people’s physical locations on an estimated virtual plane in the scene. We combine the simulated physical locations with the appearance features to track people online. The effectiveness of this method is evaluated on different dataset by extensive comparisons with state-of-the-art techniques. The experimental results reveal that the proposed method improves the original Deep SORT algorithm in both Number of Identity Switches (IDSW) and fragmented (Frag).

## Requirements

- Python 3.7
- pytorch >= 1.6.0
- torchvision >= 0.7.0
- python-opencv

## Quick start

**1. download code**

`````
git clone https://github.com/Jeasonlee313/paperdev_Phy_SORT-.git
cd paperdev_Phy_SORT-
`````

**2. Download appearance parameters last_net.pth**

```
cd deep_sort/deep/checkpoint
# download last_net.pth from https://pan.baidu.com/s/12AaJ0kYbbbdotPD_7Aqahg 
# and the password：klsf 
# to this folder
cd ../../../
```

**3. make sure the "data" and "output" folder are exist (optional)**

```
mkdir data && mkdir output
```

**4. run demo**

```
usage: python app.py [--VIDEO_PATH]
					 [--config_deepsort]
					 [--display]
					 [--frame_interval]
					 [--display_width]
					 [--display_height]
					 [--save_path]
					 [--cpu]
					 [--save_name]
					 [--detect_file]
```

parameters meaning:

- `--VIDEO_PATH` : Storage path of video data
- `--config_deepsort` : The original parameter of deepsort
- `--display` : Choose to display visualization results
- `--frame_interval` : Frame interval
- `--display_width` : Visualization window width
- `--display_height` : Visualization window height
- `--save_path` : Result output path
- `--cpu` : Use CPU only
- `--save_name` : Result file name
- `--detect_file` : Detection results of video data

The above parameters are divided into `--VIDEO_PATH`，`--save_path`，`--save_name` and `--detect_file` needs to be filled independently, and the rest have default parameters. We recommend debugging in PyCharm IDE by modifying the default parameters.



## Training the RE-ID model

We retrain the appearance model by using triplet loss. The training and test files are in the `deep_sort\deep\train.py` and `deep_sort\deep\test.py`. Here we give the test results.

| Dataset       | Rank-1 | **Rank-5** | Rank-10 | **mAP** |
| ------------- | ------ | ---------- | ------- | ------- |
| Market-1501   | 0.8625 | 0.9426     | 0.9608  | 0.7026  |
| DukeMTMC-reID | 0.7639 | 0.8712     | 0.9062  | 0.5937  |



## Thanks

My sincere thanks to my @professor Li for his novel and effective ideas and his painstaking guidance in revising our paper. And thanks to @Alexis for revising the grammar for the paper.

Special thanks to @[ZQPei](https://github.com/ZQPei) and @Zhedong Zheng, their code provides a great help for my research.

https://github.com/ZQPei/deep_sort_pytorch.git

https://github.com/layumi/Person_reID_baseline_pytorch.git

