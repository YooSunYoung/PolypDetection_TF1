from absl import app, flags
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import os
from Models.models import PolypDetectionModel
from Training import training_recipe


class Trainer:
    def __init__(self, **kwargs):
        self.train_image_path = kwargs.get("train_image")
        self.train_label_path = kwargs.get("train_label")
        self.validate = kwargs.get("validate", False)
        self.valid_image_path = kwargs.get("valid_image", None)
        self.valid_label_path = kwargs.get("valid_label", None)
        self.train_dataset = (np.load(self.train_image_path), np.load(self.train_label_path))
        if self.validate:
            if self.valid_image_path is None:
                raise AssertionError("Validation dataset path not given")
            self.val_dataset = (np.load(self.valid_image_path), np.load(self.valid_label_path))
        self.epoch = kwargs.get("epoch", 100)
        self.save_point = kwargs.get("save_point", 50)
        self.validation_point = kwargs.get("validation_point", None)
        self.batch_size = kwargs.get('batch_size', 32)
        self.val_batch_size = kwargs.get('val_batch_size', 32)
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.checkpoint_dir_path = kwargs.get("checkpoint_dir_path", "results/checkpoint/")
        self.device = kwargs.get("device", "/device:XLA_GPU:0")
        self.model = PolypDetectionModel()

    def train(self):
        train_size = len(self.train_dataset[0])
        self.batch_size = train_size if train_size < self.batch_size else self.batch_size
        self.val_batch_size = train_size if train_size < self.val_batch_size else self.val_batch_size
        batch_x, batch_y = None, None
        cur_loss, cur_val_loss = None, None
        lowest_loss = float('inf')
        with tf.device(self.device):
            with tf.Graph().as_default():
                X = tf.placeholder(tf.float32, [None, 227, 227, 3], name="input")
                Y = tf.placeholder(tf.float32, [None, 4, 4, 5], name="output")
                output_0 = self.model.get_model(X, training=True)
                loss = self.model.get_loss(y_true=Y, y_pred=output_0, train_state=True)
                global_step = tf.contrib.framework.get_or_create_global_step()
                train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=global_step)
                summary_op = tf.summary.merge_all()
                images = self.train_dataset[0]
                labels = self.train_dataset[1]
                val_size = len(self.val_dataset)
                val_images = self.val_dataset[0]
                val_labels = self.val_dataset[1]
                init = tf.global_variables_initializer()
                saver = tf.train.Saver()
                with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                    writer = tf.summary.FileWriter("../summary/", sess.graph)
                    sess.run(init)
                    cur_loss = 0
                    for i in range(1, self.epoch + 1):
                        # if lowest_loss < 20 and self.learning_rate > 1e-4:
                        #     self.learning_rate = 1e-4
                        #     train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss,
                        #                                                                    global_step=global_step)
                        batch_epoch_size = int(len(images) / self.batch_size)
                        for b_num in range(batch_epoch_size):
                            offset = b_num * self.batch_size
                            if offset + self.batch_size < train_size:
                                batch_x, batch_y = images[offset:(offset + self.batch_size)], labels[
                                                                                     offset:(offset + self.batch_size)]
                            else:
                                batch_x, batch_y = images[offset:], labels[offset:]
                            _, cur_loss, summary = sess.run([train_op, loss, summary_op],
                                                            feed_dict={X: batch_x, Y: batch_y})
                        print("training loss : ", i, cur_loss)
                        if self.validate:
                            if i % self.validation_point == 0:
                                for b_num in range(int(len(val_images) / self.val_batch_size)):
                                    val_offset = (b_num * self.val_batch_size) % (val_size - 1) if val_size > 1 else 0
                                    val_batch_x = val_images[val_offset:(val_offset + self.val_batch_size)]
                                    val_batch_y = val_labels[val_offset:(val_offset + self.val_batch_size)]
                                    cur_val_loss, val_summary = sess.run([loss, summary_op],
                                                                         feed_dict={X: val_batch_x, Y: val_batch_y})
                                if cur_val_loss < lowest_loss:
                                    lowest_loss = cur_val_loss
                                    print("current lowest validation loss : ", i, cur_val_loss)
                                    saver.save(sess, os.path.join(self.checkpoint_dir_path, "model"))
                        else:
                            if cur_loss < lowest_loss:
                                saver.save(sess, os.path.join(self.checkpoint_dir_path, "model"))

                        if i % self.save_point == 0 or i == self.epoch:
                            saver.save(sess, os.path.join(self.checkpoint_dir_path, "model"), i)


def main(_args):
    training_recipe.set_settings(flags)
    trainer = Trainer(train_image=FLAGS.train_image, train_label=FLAGS.train_label,
                      valid_image=FLAGS.valid_image, valid_label=FLAGS.valid_label, validate=FLAGS.validate,
                      epoch=FLAGS.epochs, save_point=FLAGS.save_point, validation_point=FLAGS.validation_point,
                      batch_size=FLAGS.batch_size, val_batch_size=FLAGS.val_batch_size,
                      checkpoint_dir_path=FLAGS.checkpoint_dir_path,
                      learning_rate=FLAGS.learning_rate)
    trainer.train()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
