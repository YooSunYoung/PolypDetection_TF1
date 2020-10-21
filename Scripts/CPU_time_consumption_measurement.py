from Communication.ImageReceiver import ImageReceiver
from Models.models import PolypDetectionModel
import tensorflow as tf
import time

receiver = ImageReceiver(debug_mode=False)
receiver.build_connection()
images = receiver.receive_images()
checkpoint_path = "../results/36_36_6_checkpoints/model-700"
model = PolypDetectionModel()
predictions = []
with tf.Graph().as_default():
    X = tf.placeholder(tf.float32, [None, 227, 227, 3], name="imGRBNormalize")
    output_0 = model.get_model(X, training=False)
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
print(predictions)
print("total_time: %.2f" % total_time)
