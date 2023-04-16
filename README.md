# :factory: AI Factory Detection
**Develop a machine learning model capable of detecting gold fingers and deploy it in a real-world factory environment.**

# Feature

- Develop an image or video analysis system using machine learning techniques to detect defects in gold fingers on printed circuit boards (PCBs).
- Deploy the gold finger defect detection system on Intel platforms for optimal performance and efficiency.


# Detection Example

- Original img<br>
![](README_intro_img/LUCID_6.jpg)
- Labeled img(image be rotated)<br>
![](README_intro_img/result.png)

## Table of Contents

- [Feature](#feature)
- [Maintainers](#maintainers)
- [Dependencies](#dependencies)
- [How to use](#how-to-use)
    - [Build Dockerfile](#build-dockerfile)
    - [Choose detection file](#choose-detection-file)
    - [Run container](#run-container)
- [About image process code](#about-image-process-code)
- [License](#license)

# Dependencies
1. Python 3.8.3 (default, Jul  2 2020, 16:21:59)
2. OpenCV 4.1.2.30
3. Intel® Core™ i5-7500 CPU @ 3.40GHz × 4

# How to use

## Build Dockerfile
    $ ./build.sh
## Choose detection file
open app.json and change **"input_data"** location.
    
    {
    "application": "object detection",
    "category": "sample",
    "framework": "openvino",
    "input_data": "app/golden-finger/LUCID_TRI050S-C_183700028__20220815150751168_video1.avi",
    "prim": {
        "model_json": "./app/golden-finger/yolo.json"
    },
    "app_name": "golden-finger",
    "input_type": "V4L2"
}
## Run container
    $ ./Golden-finger-detecion./docker/run.sh -f openvino

## Run openvino_demo.py
    $ cd Golden-finger-detecion
    $ python3 openvino_demo.py -c app/golden-finger/app.json
    

# About image process code 

## 1. Grab PCB image in original image
* Original img<br>
![](README_intro_img/LUCID_6.jpg)


* Green in the range(image be rotated)<br>
![](README_intro_img/pcb_mask_with_noises.png)

* Remove noises on the mask<br>
![](README_intro_img/remove_noise_mask.png)

* Calculate counter area and find PCB location<br>
![](README_intro_img/pcb_mask1.png)

* Cut pcb on original image<br>
![](README_intro_img/cut_pcb.png)

### 2. Grab gold finger in PCB image
- original pcb image<br>
![](README_intro_img/cut_pcb.png)

- get gold finger mask<br>
![](README_intro_img/finger_mask.png)

- Cut gold fonger on each pcb<br>
![](README_intro_img/cut_finger.png)


## 3. Cut photo into equal parts
- original  gold fonger image<br>
![](README_intro_img/cut_finger.png)

- To be trained by yolov4, we cut and resize those images. 
In this task, we will be resizing the images to a size of 608x608 pixels..<br>
![](README_intro_img/tar_back1.png)


# Reference

# Maintainers
[@Johnny-Hu-406](https://github.com/Johnny-Hu-406)

# License
[MIT](LICENSE) @Johnny-Hu-406
