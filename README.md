# :factory: AI Factory Detection
**Train a gold finger detection model and deploy on real factory environment.**
<br>

## Feature
---
- Detect Gold finger decfct on a PCB image or video
- Deployment to Intel platforms
<br>

## Detection Example
---
![](https://github.com/Johnny-Hu-intern/ai-factory-detection/blob/aa55a8925498b69ec1c134d72e82847ca83e8c95/README_intro_img/LUCID_6.jpg)

- Original img

![](https://github.com/Johnny-Hu-intern/ai-factory-detection/blob/aa55a8925498b69ec1c134d72e82847ca83e8c95/README_intro_img/result.png)

- Labeled img
<br>

## Dependencies
---
1.Python 3.8.3 (default, Jul  2 2020, 16:21:59)

2.OpenCV 4.1.2.30

3.Intel® Core™ i5-7500 CPU @ 3.40GHz × 4
<br>

## How to use
---
### Build Dockerfile
    ./build.sh
### Choose detection file
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
### Run container
    ./docker/run.sh -f openvino
    python3 openvino_demo.py -c app/golden-finger/app.json
<br>

## About image process code 
---
#### Grab PCB image in original image
![](https://github.com/Johnny-Hu-intern/ai-factory-detection/blob/aa55a8925498b69ec1c134d72e82847ca83e8c95/README_intro_img/LUCID_6.jpg)

- Original img

<!-- ![](https://github.com/Johnny-Hu-intern/ai-factory-detection/blob/aa55a8925498b69ec1c134d72e82847ca83e8c95/README_intro_img/remove_noise_mask_screenshot_17.08.2022.png) -->
- remove noises on the msak
#### Grab gold finger in PCB image
#### Cut photo into equal parts
<br>

## Reference
---
