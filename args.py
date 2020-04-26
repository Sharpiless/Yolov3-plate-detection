# coding: utf-8
# This file contains the parameter used in train.py

from __future__ import division, print_function

from utils.misc_utils import parse_anchors, read_class_names
import math


restore_path = './data/darknet_weights/yolov3.ckpt'  # The path of the weights to restore.
save_dir = './checkpoint/'  # The directory of the weights to save.
anchor_path = './data/yolo_anchors.txt'  # The path of the anchor txt file.
class_name_path = './data/mydata.names'  # The path of the class names.

### Training releated numbers
img_size = [17*32, 29*32]  # Images will be resized to `img_size` and fed to the network, size format: [width, height]

### tf.data parameters
num_threads = 10  # Number of threads for image processing used in tf.data pipeline.
prefetech_buffer = 5  # Prefetech_buffer used in tf.data pipeline.

### some constants in validation
# nms
nms_threshold = 0.45  # iou threshold in nms operation
score_threshold = 0.01  # threshold of the probability of the classes in nms operation, i.e. score = pred_confs * pred_probs. set lower for higher recall.
nms_topk = 150  # keep at most nms_topk outputs after nms
# mAP eval
eval_threshold = 0.5  # the iou threshold applied in mAP evaluation
use_voc_07_metric = False  # whether to use voc 2007 evaluation metric, i.e. the 11-point metric

### parse some params
anchors = parse_anchors(anchor_path)
classes = read_class_names(class_name_path)
class_num = len(classes)
