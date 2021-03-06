import os
import cv2
import time
import numpy as np
import tensorflow as tf
from Models.models import PolypDetectionModel
import logging
import xml.etree.ElementTree


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
                time1 = time.time()
                prediction = sess.run(output_0, feed_dict={X: image_rgb})
                time2 = time.time()
                predictions.append(prediction)
                total_time = time2 - time1
    print("total_time: %.2f" % total_time)
    return predictions


def extend_image(img, width=500, height=500, color=(0, 0, 0)):
    original_height, original_width, channel = img.shape
    extended = np.full((width, height, channel), color, dtype=np.uint8)
    center_offset_x, center_offset_y = int((width - original_width) // 2), int((height - original_height) // 2)
    extended[center_offset_y:center_offset_y+original_height, center_offset_x:center_offset_x+original_width] = img
    return extended


def draw_outputs(img, outputs, class_names, label, color, extended_image_width=400, extended_image_height=400,
                 original_image_width=227, original_image_height=227):
    offsets = tuple([int((extended_image_width - original_image_width) / 2),
                     int((extended_image_height - original_image_height) / 2)])
    if extended_image_width > img.shape[0] and extended_image_height > img.shape[1]:
        img = extend_image(img, extended_image_width, extended_image_height)
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip([original_image_width, original_image_height])
    for i in range(1):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        if x2y2[0] <= 0 or x2y2[1] <= 0:
            img = cv2.putText(img, "No Polyp, {}".format(label),
                              (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            return img
        x1y1 = tuple([a + b for a, b in zip(offsets, x1y1)])
        x2y2 = tuple([a + b for a, b in zip(offsets, x2y2)])
        img = cv2.rectangle(img, x1y1, x2y2, color, 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        label_pos = (x1y1[0], x2y2[1])
        img = cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
    return img


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


if __name__=="__main__":
    multiple_checkpoints = 0
    multiple_images = 0
    ground_truth = 0
    checkpoint_paths = []
    image_paths = []
    if not multiple_checkpoints:
        checkpoint_path = "../results/36_36_6_checkpoints/model-700"
        checkpoint_paths.append(checkpoint_path)
    else:
        checkpoint_dir = "../results/checkpoints/"
        checkpoint_files = ["model-100", "model-200", "model-300", "model-500"]
        checkpoint_paths = [os.path.join(checkpoint_dir, file) for file in checkpoint_files]

    if not multiple_images:
        image_path = "../data/PolypImages_train/028.jpg"
        image_paths.append(image_path)
    else:
        #image_dir = "../data/PolypImages_train/"
        image_dir = "../data/PolypImages_test"
        image_files = os.listdir(image_dir)
        image_files = [x for x in image_files if x.endswith(".jpg")]
        image_paths = [os.path.join(image_dir, file) for file in image_files]

    result_directory = mkdir("inference_result")
    class_names = ["Polyp"]
    color_list = [(0, 130, 153), (171, 242, 0), (29, 219, 22), (0, 216, 255), (0, 84, 255)]

    images = []
    for image_path in image_paths:
        file_name = image_path.split("/")[-1]
        image, img_rgb = get_single_image(image_path=image_path)
        if ground_truth:
            x1y1, x2y2 = get_ground_truth(image_path.replace(".jpg",".xml"))
            image = cv2.rectangle(image, x1y1, x2y2, (0, 0, 255), 4)
            image = cv2.putText(image, "Ground Truth", x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 4)
        images.append([image, img_rgb, file_name])

    predictions = []
    for ich, checkpoint_path in enumerate(checkpoint_paths):
        predictions.append(get_prediction(checkpoint_path, images))

    for iimage, (image, image_rbg, file_name) in enumerate(images):
        for ich in range(len(checkpoint_paths)):
            label = checkpoint_paths[ich].split("/")[-1]
            prediction = predictions[ich][iimage]
            # np.save("DNNOutput_PC", predictions)
            boxes, scores, classes, nums = prediction
            if scores[0][0] >= 2.5: print(file_name)
            logging.info('Image {} detections: '.format(file_name))
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                   np.array(scores[0][i]),
                                                   np.array(boxes[0][i])))
            image = draw_outputs(image, outputs=prediction, class_names=class_names,
                                 label=label, color=color_list[ich])
        cv2.imwrite(os.path.join(result_directory, "output_"+file_name), image)
