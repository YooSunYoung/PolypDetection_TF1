import os
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from Utils.common_functions import make_and_clean_dir
from Models.models import PolypDetectionModel
from Training import training_recipe
training_recipe.set_settings(flags)

IMAGE_SIZE = 28

def main(_args):
    p = PolypDetectionModel(data_path=FLAGS.dataset)
    input_images = p.get_dataset()
    epoch = FLAGS.epochs
    save_point = FLAGS.save_points
    learning_rate = FLAGS.learning_rate
    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, [None, 227, 227, 3], name="imGRBNormalize")
        Y = tf.placeholder(tf.float32, [None, 4, 4, 5], name="output")
        output_0 = p.get_model(X, training=True)
        loss = p.get_loss(y_true=Y, y_pred=output_0, train_state=True)
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        summary_op = tf.summary.merge_all()
        images = input_images[0]
        val_images = input_images[0]
        labels = input_images[1]
        val_labels = input_images[1]
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        batch_size = FLAGS.batch_size
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            writer = tf.summary.FileWriter("../summary/", sess.graph)
            sess.run(init)
            for i in range(1, epoch+1):
                for b_num in range(int(len(images)/batch_size)):
                    offset = (b_num * batch_size) % (len(images) - 1) if len(images) > 1 else 0
                    batch_x, batch_y = images[offset:(offset + 1)], labels[offset:(offset +1)]
                    _, cur_loss, summary = sess.run([train_op, loss, summary_op],
                                                    feed_dict={X: batch_x, Y: batch_y})
                print("testing loss : ",i, cur_loss)
                if i%save_point==0 or i==epoch:
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir_path, "model"), i)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass