from Communication.ImageReceiver import ImageReceiver
from Models.models import PolypDetectionModel
import tensorflow as tf
import time
import numpy as np
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def process_output(result):
    grid_size_width = 4
    grid_size_height = 4
    image_size_width = 227
    image_size_height = 227
    cx = result[0]
    cy = result[1]
    w = result[2]
    h = result[3]
    num_box = 3
    num_entry = 5
    x1 = cx - w / 2 * 227
    x2 = cx + w / 2 * 227
    y1 = cy - h / 2 * 227
    y2 = cy + h / 2 * 227
    # x1 = result[0]*227
    # y1 = result[1]*227
    # x2 = result[2]*227
    # y2 = result[3]*227
    return [x1, x2, y1, y2, result[4]]


if __name__=="__main__":
    receiver = ImageReceiver(debug_mode=False, port=6004)
    receiver.build_connection()
    images = receiver.receive_images()
    num_test = 1
    nn_times = []
    latency_nn_times = []
    predictions = None
    for i_test in range(num_test):
        time_stamp_image_received = time.time()
        images = [np.array(image).reshape((227, 227, 3)) for image in images]
        for iimgs, imgs in enumerate(images):
            img_shape = imgs.shape
            imRGB = np.zeros((img_shape[0], img_shape[1], img_shape[2]), dtype=float)
            for i in range(imgs.shape[0]):
                for j in range(imgs.shape[1]):
                    imRGB[i, j, 0] = imgs[i, j, 2]
                    imRGB[i, j, 1] = imgs[i, j, 1]
                    imRGB[i, j, 2] = imgs[i, j, 0]
            imRGB = imRGB/255.0
            imRGB = np.array([imRGB])
            images[iimgs] = imRGB
        checkpoint_path = "../../results/old_results/48_48_6_checkpoints/model-700"
        model = PolypDetectionModel()
        predictions = []
        single_time_consumption = 0.0
        with tf.Graph().as_default():
            X = tf.placeholder(tf.float32, [None, 227, 227, 3], name="imGRBNormalize")
            output_0 = model.get_model(X, training=False)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                sess.run(init)
                saver.restore(sess, checkpoint_path)
                time_stamp_1 = time.time()
                for image_rgb in images:
                    prediction = sess.run(output_0, feed_dict={X: image_rgb})
                    predictions.append(prediction)
                time_stamp_2 = time.time()

        results = []
        for prediction in predictions:
            boxes, objectness, classes, nums = prediction
            #result = prediction[0][0][0]
            #result = np.append(result, prediction[1][0][0])
            result = boxes[0][0]
            result = np.append(result, objectness[0][0])
            result = process_output(result)
            results.append(result)


        time_stamp_image_sent = time.time()
        total_nn_time_consumption = time_stamp_2 - time_stamp_1
        total_time_consumption = time_stamp_image_sent - time_stamp_image_received

        nn_times.append(total_nn_time_consumption)
        latency_nn_times.append(total_time_consumption)
        del model

    for result in results:
        receiver.send_array(result)


    print("Processing done with {} images".format(len(predictions)))
    print("Total NN time")
    for total_nn_time_consumption in nn_times:
        print("%.5f" % total_nn_time_consumption)
    print("Total time")
    for total_time_consumption in latency_nn_times:
        print("%.5f" % total_time_consumption)
    receiver.close_connection()

    save_result = "%.5f %.5f\n" % (total_nn_time_consumption, total_time_consumption)
    result_txt_file = sys.argv[1]
    with open(result_txt_file, 'a') as file:
        file.write(save_result)
