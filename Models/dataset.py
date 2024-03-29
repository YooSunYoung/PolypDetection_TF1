import os
import logging
import cv2
import xml.etree.ElementTree
import numpy as np
from Models.config import configuration as config

gridSize = config.get("grid_size")
nboxes = config.get("n_boxes")
imageW = config.get("image_width")
imageH = config.get("image_height")


def transform_targets(y_true):
    label = np.zeros((gridSize, gridSize, 5))
    x1 = y_true[0]  # x min
    y1 = y_true[1]  # y min
    x2 = y_true[2]  # x max
    y2 = y_true[3]  # y max
    center_x = (x2 + x1) / 2
    center_y = (y2 + y1) / 2
    w = x2 - x1
    h = y2 - y1
    grid_x = int(center_x * gridSize)
    grid_y = int(center_y * gridSize)
    label[grid_y, grid_x] = [1, center_x, center_y, w, h]
    return label


class DataSet:
    def __init__(self, data_path="../data/", valid_data_path="../data/", test_data_path="../data/"):
        self.data_path = data_path
        self.valid_data_path = valid_data_path
        self.test_data_path = test_data_path

    def load_data(self, data_path=None, training=True):
        if data_path is None: data_path = self.data_path
        logging.log(logging.INFO, "Scraping dataset from {}".format(data_path))
        images = []
        norm_images, labels = [], []
        files = os.listdir(data_path)
        for file in files:
            if '.jpg' in file:
                images.append([os.path.join(data_path, file), 1])
        for image in images:
            im = cv2.imread(image[0], cv2.IMREAD_COLOR)
            im = cv2.resize(im, (227, 227))
            im = im.reshape((227, 227, 3))
            im_shape = im.shape
            # BGR to RGB
            im_rgb = np.zeros((im_shape[0], im_shape[1], im_shape[2]), dtype=float)
            for i in range(im_shape[0]):
                for j in range(im_shape[1]):
                    im_rgb[i, j, 0] = im[i, j, 2]
                    im_rgb[i, j, 1] = im[i, j, 1]
                    im_rgb[i, j, 2] = im[i, j, 0]
            im_rgb_normalized = im_rgb / 255.0
            # parsing xml
            xml_name = image[0].replace('jpg', 'xml')
            meta = None
            if os.path.isfile(xml_name):
                meta = xml.etree.ElementTree.parse(xml_name).getroot()
            bndbox = np.zeros(4)
            if meta is not None:
                obj = meta.find('object')
                if obj is not None:
                    box = obj.find('bndbox')
                    if box is not None:
                        bndbox[0] = int(box.find('xmin').text.split('.')[0])
                        bndbox[1] = int(box.find('ymin').text.split('.')[0])
                        bndbox[2] = int(box.find('xmax').text.split('.')[0])
                        bndbox[3] = int(box.find('ymax').text.split('.')[0])
            norm_images.append(im_rgb_normalized)
            if training:
                bndbox_normalized = bndbox / 227
                y_true = transform_targets(bndbox_normalized)
                labels.append(y_true)
            else:
                labels.append(bndbox)

        return np.array(norm_images), np.array(labels)

    def load_train_data(self, data_path=None):
        if data_path is None:
            data_path = self.data_path
            logging.log(logging.INFO, "Training dataset from directory {}".format(data_path))
        else:
            if data_path is not self.data_path:
                self.data_path = data_path
                logging.log(logging.INFO, "Training dataset from directory {}".format(data_path))
        return self.load_data(self.data_path)

    def load_valid_data(self, data_path=None):
        if data_path is None:
            data_path = self.valid_data_path
            logging.log(logging.INFO, "Validation dataset from directory {}".format(data_path))
        else:
            if data_path is not self.valid_data_path:
                self.valid_data_path = data_path
                logging.log(logging.INFO, "Validation dataset from directory {}".format(data_path))
        return self.load_data(data_path)

    def load_test_data(self, data_path=None):
        if data_path is None:
            data_path = self.test_data_path
            logging.log(logging.INFO, "Validation dataset from directory {}".format(data_path))
        else:
            if data_path is not self.test_data_path:
                self.test_data_path = data_path
                logging.log(logging.INFO, "Validation dataset from directory {}".format(data_path))
        return self.load_data(data_path, training=False)


if __name__ == "__main__":
    dataset = DataSet(data_path='../data/PolypImages_train/',
                      valid_data_path='../data/PolypImages_valid/',
                      test_data_path='../data/PolypImages_test/')

    training_image, training_label = dataset.load_train_data()
    valid_image, valid_label = dataset.load_valid_data()
    test_image, test_label = dataset.load_test_data()

    with open('../data/train_image.npy', 'wb') as f:
        np.save(f, np.array(training_image))
    with open('../data/train_label.npy', 'wb') as f:
        np.save(f, np.array(training_label))

    with open('../data/valid_image.npy', 'wb') as f:
        np.save(f, np.array(valid_image))
    with open('../data/valid_label.npy', 'wb') as f:
        np.save(f, np.array(valid_label))

    with open('../data/test_image.npy', 'wb') as f:
        np.save(f, np.array(test_image))
    with open('../data/test_label.npy', 'wb') as f:
        np.save(f, np.array(test_label))
