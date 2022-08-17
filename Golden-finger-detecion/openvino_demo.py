#!/usr/bin/python3
# -*- coding: utf-8 -*-
from ast import parse
from random import normalvariate
import cv2
import sys
import logging
from api.utils import Json, Draw
from api.utils.logger import config_logger
from api.real_class_test import Defect_detect
import argparse
import numpy as np

def main(args):
    # Instantiation
    json = Json()
    draw = Draw()

    # Get to relative parameter from first json
    custom_cfg = json.read_json(args.config)
    dev_cfg = [custom_cfg[key] for key in custom_cfg.keys() if "prim" in key]
    # Summarized previous dictionary and get to relative parameter from secondary json  
    for prim_ind in range(len(dev_cfg)):
        # Input_data append to dev_cfg
        dev_cfg[prim_ind].update({"input_data":custom_cfg['input_data']})
        dev_cfg[prim_ind].update(json.read_json(dev_cfg[prim_ind]['model_json']))
        # Check is openvino and start processes
        if custom_cfg['framework'] == 'openvino':
        # ---------------------------Check model architecture-------------------------------------------------------
            if 'obj' in dev_cfg[prim_ind]['tag']:
                from api.obj import ObjectDetection as trg
                
        # ---------------------------Load model and initial pipeline--------------------------------------------------------------------
            trg = trg()
            model, color_palette = trg.load_model(dev_cfg[prim_ind])

        # ---------------------------Check input is camera or image and initial frame id/show id----------------------------------------
            from api.common.images_capture import open_images_capture
            cap = open_images_capture(dev_cfg[prim_ind]['input_data'], dev_cfg[prim_ind]['openvino']['loop'])

        # ---------------------------Inference---------------------------------------------------------------------------------------------
            logging.info('Starting inference...')
            print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
            while True:
                frame = cap.read()

                name = "img"
                detections = {}

                rotation = np.rot90(frame)
                detecter = Defect_detect(img=rotation, filename="pcb")
                loc, frame_list = detecter.detect()

                if frame_list == 0:
                    continue 
                for cut_frame_name in frame_list:
                    cut_frame = frame_list[cut_frame_name]
                    info = trg.inference(model, cut_frame, dev_cfg[prim_ind])
                    if info["detections"] != [] : 
                        detections[name]={"name":cut_frame_name, "loc":loc, "detections":info["detections"]}
        # ---------------------------Drawing detecter to information-----------------------------------------------------------------------
                if detections != {}:
                    rotation = draw.draw_detections(rotation, detections, color_palette)
        # ---------------------------Show--------------------------------------------------------------------------------------------------  
                # Resize image to show easily
                cv2.resizeWindow('Detection', 608, 608)
                cv2.imshow("Detection", rotation)
                key = cv2.waitKey(1)
                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break

                

if __name__ == '__main__':
    config_logger('./VINO.log', 'w', "info")
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help = "The path of application config")
    args = parser.parse_args()

    sys.exit(main(args) or 0)