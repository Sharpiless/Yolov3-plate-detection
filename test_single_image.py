# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import args as cfg
import argparse
import cv2
import os

from utils.misc_utils import read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3

new_size = cfg.img_size

resize = True

class_name_path = './data/mydata.names'

restore_path = cfg.restore_path

classes = read_class_names(class_name_path)

num_class = len(classes)

color_table = get_color_table(num_class)


def demo(input_image):

    img_ori = cv2.imread(input_image)
    if resize:
        img, resize_ratio, dw, dh = letterbox_resize(
            img_ori, new_size[0], new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] - 127.5

    boxes_, scores_, labels_ = sess.run(
        [boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image
    if letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori/float(new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(new_size[1]))

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)

    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[
            i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])

    cv2.imshow('result', img_ori)
    cv2.waitKey(0)


if __name__ == '__main__':

    test_path = './demo_images'

    im_names = os.listdir(test_path)

    with tf.Session() as sess:

        input_data = tf.placeholder(
            tf.float32, [1, new_size[1], new_size[0], 3], name='input_data')
        yolo_model = yolov3(num_class, cfg.anchors)
        print(num_class)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(
            pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(
            pred_boxes, pred_scores, num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(sess, restore_path)

        for name in im_names:

            path = os.path.join(test_path, name)
            demo(path)

        cv2.destroyAllWindows()
