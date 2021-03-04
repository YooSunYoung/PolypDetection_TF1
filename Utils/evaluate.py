import os
import numpy as np
import tensorflow as tf
from Models.models import PolypDetectionModel
import logging
from absl import app, logging

import matplotlib.pyplot as plt
import pandas as pd


def iou(box1, box2):
    '''
    box:[x1, y1, x2, y2]
    '''
    in_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou


def get_prediction(checkpoint_path, images):
    p = PolypDetectionModel()
    predictions = []
    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, [None, 227, 227, 3], name="input")
        output_0 = p.get_model(X, training=False)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(init)
            saver.restore(sess, checkpoint_path)
            for image, image_rgb, file_name in images:
                prediction = sess.run(output_0, feed_dict={X: image_rgb})
                predictions.append(prediction)
    return predictions


def extend_image(img, width=500, height=500, color=(0, 0, 0)):
    original_height, original_width, channel = img.shape
    extended = np.full((width, height, channel), color, dtype=np.uint8)
    center_offset_x, center_offset_y = int((width - original_width) // 2), int((height - original_height) // 2)
    extended[center_offset_y:center_offset_y+original_height, center_offset_x:center_offset_x+original_width] = img
    return extended


def get_bounding_box(boxes):
    wh = 227
    x1y1 = tuple((np.array(boxes[0][0][0:2]) * wh).astype(np.int32))
    x2y2 = tuple((np.array(boxes[0][0][2:4]) * wh).astype(np.int32))
    if x2y2[0] <= 0 or x2y2[1] <= 0:
        return None
    return x1y1, x2y2


def mkdir(directory_path, i=0):
    if os.path.exists(directory_path):
        if len(os.listdir(directory_path)) is 0:
            logging.info("Directory {} already exists but empty. Will use this directory.".format(directory_path))
            return directory_path
        else:
            logging.info("Directory {} already exists.".format(directory_path))
            if i is 0:
                directory_path = directory_path + "_" + str(i+1)
            else:
                directory_path = directory_path.replace("_" + str(i), "_" + str(i+1))
            i += 1
            directory_path = mkdir(directory_path, i)
            return directory_path
    else:
        os.mkdir(directory_path)
    return directory_path


def main(_agrs):
    checkpoint_path = "../results/model"
    test_image_path = "../data/test_image.npy"
    test_label_path = "../data/test_label.npy"
    test_image = np.load(test_image_path)
    test_label = np.load(test_label_path)

    polyp_images_dict = {}
    for iimg, (image, label) in enumerate(zip(test_image, test_label)):
        properties = {}
        x1y1, x2y2 = label[:2], label[2:4]
        properties["image_rgb"] = [image]
        properties["ground_truth"] = [x1y1, x2y2]
        properties["score"] = None
        properties["bndbox"] = None
        polyp_images_dict[iimg] = properties
    # images without any polyp should be included in evaluation

    p = PolypDetectionModel()
    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, [None, 227, 227, 3], name="input")
        output_0 = p.get_model(X, training=False)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(init)
            saver.restore(sess, checkpoint_path)
            for image in polyp_images_dict.keys():
                prediction = sess.run(output_0, feed_dict={X: polyp_images_dict[image]["image_rgb"]})
                boxes, objectness, classes, nums = prediction
                coordinates = get_bounding_box(boxes)
                if coordinates is not None:
                    polyp_images_dict[image]["bndbox"] = [coordinates[0], coordinates[1]]
                polyp_images_dict[image]["score"] = objectness[0][0]

    for image in polyp_images_dict.keys():
        # This part is very important! Note that we only consider one bounding box outputted by DNN
        properties = polyp_images_dict[image]
        bndbox = properties["bndbox"]
        if bndbox is not None:
            ground_truth = properties["ground_truth"]
            ground_truth = tuple(ground_truth[0]) + tuple(ground_truth[1])
            bndbox = bndbox[0] + bndbox[1]
            iouRs = iou(ground_truth, bndbox)
            properties["iou"] = iouRs
        else: properties["iou"] = 0

    tmp_ax = None
    for iou_threshold in np.arange(0.1, 0.5, 0.05):
        # The AP is calcluated in this part
        recall = 0.0
        precision = 1.0
        sumArea = 0.0
        precisionList = []
        recallList = []
        for scoreThreshold in np.arange(0.98, -0.1, -0.1):  # score is actually the object probability.
            TN = 0  # True negative
            FP = 0  # False positive
            TP = 0  # True positive
            FN = 0  # False negative
            for image in polyp_images_dict.keys():
                properties = polyp_images_dict[image]
                score = properties["score"]
                if score >= scoreThreshold:
                    iouRs = properties["iou"]
                    if iouRs > iou_threshold:
                        TP = TP + 1
                    else:
                        FP = FP + 1
                else:
                    FN = FN+1
                    TN = TN+1

            previousRecall = recall
            previousPrecision = precision
            if (TP + FP) == 0:
                precision = 0
            else:
                precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            deltaRecall = np.fabs(recall - previousRecall)
            deltaArea = deltaRecall * (precision + previousPrecision) / 2.0 # calculate the area.
            sumArea = sumArea + deltaArea #sumArea is the AP that we want
            print("sumArea:{:.3f}, precision:{:.3f},recall:{:.3f},deltaRecall:{:.3f}".format(sumArea, precision, recall, deltaRecall))

            precisionList.append(precision)
            recallList.append(recall)
        precision_recall_df = pd.DataFrame({'precision': precisionList, 'recall': recallList})
        if tmp_ax is None:
            tmp_ax = precision_recall_df.plot(kind='line', x='recall', y='precision')
        else:
            precision_recall_df.plot(kind='line', x='recall', y='precision', ax=tmp_ax)
    threshold_label = np.arange(0.1, 0.5, 0.05)
    threshold_label = ['{:.2f}'.format(x) for x in threshold_label]
    tmp_ax.legend(threshold_label)
    plt.ylabel("Precision", fontsize=20)
    plt.xlabel("Recall", fontsize=20)
    plt.show()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass