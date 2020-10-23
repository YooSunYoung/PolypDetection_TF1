from Communication.ImageReceiver import ImageReceiver
from Models.models import PolypDetectionModel
import tensorflow as tf
import time
import numpy as np

receiver = ImageReceiver(debug_mode=False, port=6004)
receiver.build_connection()
images = receiver.receive_images()
images = [np.array(image).reshape((1,227,227,3)) for image in images]
checkpoint_path = "../results/48_48_6_checkpoints/model-700"
model = PolypDetectionModel()
predictions = []
single_time_consumption = 0.0
time_stamp_0 = time.time()
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
time_stamp_3 = time.time()

total_nn_time_consumption = time_stamp_2 - time_stamp_1
single_time_consumption = total_nn_time_consumption / len(predictions)
total_time_consumption = time_stamp_3 - time_stamp_0

print("Processing done with {} images".format(len(predictions)))
#for prediction in predictions:
#    result = prediction[0][0][0]
#    result = np.append(result, prediction[1][0][0])
#    print(result)

print("single average time: %.5f" % single_time_consumption)
print("total nn time: %.5f" % total_nn_time_consumption)
print("total time: %.5f" % total_time_consumption)
receiver.close_connection()

