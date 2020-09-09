import os
import cv2
import numpy as np
import tensorflow as tf
from Models.models import PolypDetectionModel
import logging
import xml.etree.ElementTree
from absl import app, flags, logging
from absl.flags import FLAGS


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


def get_single_image(image_path):
    imgs = cv2.imread(image_path, cv2.IMREAD_COLOR)
    imgs = cv2.resize(imgs, (227, 227))
    imgs = imgs.reshape((227, 227, 3))
    # BGR to RGB
    img_shape = imgs.shape
    imRGB = np.zeros((img_shape[0], img_shape[1], img_shape[2]), dtype=float)
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            imRGB[i, j, 0] = imgs[i, j, 2]
            imRGB[i, j, 1] = imgs[i, j, 1]
            imRGB[i, j, 2] = imgs[i, j, 0]
    imRGB = imRGB/255.0
    imRGB = np.array([imRGB])
    return imgs, imRGB


def get_images(image_dir):
    file_list = os.listdir(image_dir)
    image_list = []
    for file in file_list:
        image, img_rgb = get_single_image(file)
        image_list.append([image, img_rgb])
    return image_list


def get_prediction(checkpoint_path, images):
    p = PolypDetectionModel()
    predictions = []
    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, [None, 227, 227, 3], name="imGRBNormalize")
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


def get_ground_truth(xml_path):
    meta = xml.etree.ElementTree.parse(xml_path).getroot()
    x1y1, x2y2 = np.zeros(2, dtype=int), np.zeros(2, dtype=int)
    if meta is not None:
        obj = meta.find('object')
        if obj is not None:
            box = obj.find('bndbox')
            if box is not None:
                x1y1[0] = int(box.find('xmin').text.split('.')[0])
                x1y1[1] = int(box.find('ymin').text.split('.')[0])
                x2y2[0] = int(box.find('xmax').text.split('.')[0])
                x2y2[1] = int(box.find('ymax').text.split('.')[0])
    return tuple(x1y1), tuple(x2y2)


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


def get_image_file_paths(dir):
    image_files = os.listdir(dir)
    image_files = [x for x in image_files if x.endswith(".jpg")]
    image_paths = [os.path.join(dir, file) for file in image_files]
    return image_paths


def main(_agrs):
    checkpoint_path = "../results/checkpoints/model-400"
    polyp_image_dir = "../data/PolypImages_test/"
    no_polyp_image_dir = "../data/WithoutPolypImages/"

    polyp_image_paths = get_image_file_paths(polyp_image_dir)
    no_polyp_image_paths = get_image_file_paths(no_polyp_image_dir)

    polyp_images_dict = {}
    for image_path in polyp_image_paths:
        properties = {}
        file_name = image_path.split("/")[-1]
        image, img_rgb = get_single_image(image_path=image_path)
        x1y1, x2y2 = get_ground_truth(image_path.replace(".jpg", ".xml"))
        properties["file_name"] = file_name
        properties["ground_truth"] = [x1y1, x2y2]
        properties["image_rgb"] = img_rgb
        properties["score"] = None
        properties["bndbox"] = None
        polyp_images_dict[file_name] = properties
        #image = cv2.rectangle(image, x1y1, x2y2, (0, 0, 255), 4)
        #image = cv2.putText(image, "Ground Truth", x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 4)

    no_polyp_images_dict = {}
    for image_path in no_polyp_image_paths:
        properties = {}
        file_name = image_path.split("/")[-1]
        image, img_rgb = get_single_image(image_path=image_path)
        properties["file_name"] = file_name
        properties["image_rgb"] = img_rgb
        properties["score"] = None
        properties["bndbox"] = None
        no_polyp_images_dict[file_name] = properties

    p = PolypDetectionModel()
    predictions = []
    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, [None, 227, 227, 3], name="imGRBNormalize")
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


            for image in no_polyp_images_dict.keys():
                prediction = sess.run(output_0, feed_dict={X: no_polyp_images_dict[image]["image_rgb"]})
                boxes, objectness, classes, nums = prediction
                coordinates = get_bounding_box(boxes)
                if coordinates is not None:
                    no_polyp_images_dict[image]["bndbox"] = [coordinates[0], coordinates[1]]
                no_polyp_images_dict[image]["score"] = objectness[0][0]

    #The AP is calcluated in this part
    iouThreshold=0.5    # this parameter is important and is set to be 0.5 now
    previousRecall = 0.0
    previousPrecision = 1.0
    recall = 0.0
    precision = 1.0
    sumArea = 0.0
    precisionList = []
    recallList = []

    for image in polyp_images_dict.keys():
        # This part is very important! Note that we only consider one bounding box outputted by DNN
        iouRs = 0.0
        properties = polyp_images_dict[image]
        score = properties["score"]
        bndbox = properties["bndbox"]
        if bndbox is not None:
            ground_truth = properties["ground_truth"]
            ground_truth = ground_truth[0] + ground_truth[1]
            bndbox = bndbox[0] + bndbox[1]
            iouRs = iou(ground_truth, bndbox)
            properties["iou"] = iouRs
        else: properties["iou"] = 0

    default_TN = 0
    for image in no_polyp_images_dict.keys():
        bndbox = no_polyp_images_dict[image]["bndbox"]
        if bndbox is None:
            default_TN += 1

    for scoreThreshold in np.arange(0.98, -0.1, -0.1):  #score is actually the object probability.
        TN = default_TN  #True negative
        FP = 0  #False positive
        for image in no_polyp_images_dict.keys():
            bndbox = no_polyp_images_dict[image]["bndbox"]
            if bndbox is None: pass
            score = no_polyp_images_dict[image]["score"]
            if score < scoreThreshold:
                TN = TN + 1
            else:
                FP = FP + 1
        TP = 0  #True positive
        FN = 0  #False negative
        for image in polyp_images_dict.keys():
            properties = polyp_images_dict[image]
            score = properties["score"]
            if score >= scoreThreshold:
                iouRs = properties["iou"]
                if iouRs > iouThreshold:
                    TP = TP + 1
                else:
                    FP = FP + 1
            else:
                FN = FN+1

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

    #Save the data for futher processing, such as in Matlab
    f = open("APData_iouThreshold_{:.2f}.txt".format(iouThreshold), "w")
    for i in range(0, len(recallList)):
        precision = precisionList[i]
        recall = recallList[i]
        f.write("{:.3f},{:.3f}\n".format(precision, recall))
    f.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass