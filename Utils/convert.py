import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from Models.models import PolypDetectionModel
from Utils.detect import get_single_image

p = PolypDetectionModel()
image = get_single_image("../data/PolypImages/001.jpg")
checkpoint_path = "../results/model"
with tf.Graph().as_default():
    X = tf.placeholder(tf.float32, [None, 227, 227, 3], name="imGRBNormalize")
    output_0 = p.get_model(X, training=False)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)
        saver.restore(sess, checkpoint_path)
        minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ["FinalOutput"])
        tf.io.write_graph(minimal_graph, '.', 'model.pb', as_text=False)
        tf.io.write_graph(minimal_graph, '.', 'model.txt', as_text=True)
