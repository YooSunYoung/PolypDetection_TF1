from Models.dataset import PolypDataset as polyp_dataset
import tensorflow as tf
from tensorflow.nn import conv2d
from tensorflow_core.python.eager.wrap_function import VariableHolder
import tensorflow.keras.backend as K

def Conv2D(name, x, out_channel=3, kernel_shape=3, stride=1, padding="SAME"):
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[3]
    kernel_shape = [kernel_shape, kernel_shape]
    stride = [stride, stride, stride, stride]
    filter_shape = kernel_shape + [in_channel, out_channel]
    W_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                            uniform=False,
                                                            seed=None,
                                                            dtype=tf.float32)
    #b_init = tf.constant_initializer()
    W = tf.get_variable('W'+name, filter_shape, initializer=W_init)
    #b = tf.get_variable('b'+name, [out_channel], initializer=b_init)
    conv = tf.nn.conv2d(x, W, stride, padding, name=name)
    activation = tf.identity
    #ret = activation(tf.nn.bias_add(conv, b, data_format="NHWC"), name=name)
    ret = activation(conv, name=name)
    ret.variables = VariableHolder(W)
    #ret.variables.b = b
    return ret


def MaxPooling(name, x, shape=3, stride=None, padding='VALID', data_format="channels_last"):
    if stride is None: stride = shape
    ret = tf.layers.max_pooling2d(x, shape, stride, padding, data_format)
    return tf.identity(ret, name=name)


def halffire(name, x, num_squeeze_filters=12, num_expand_3x3_filters=12, skip=0):
    out_squeeze = Conv2D('squeeze_conv_'+name, x, out_channel=num_squeeze_filters, kernel_shape=1, stride=1, padding='SAME')
    out_expand_3X3 = Conv2D('expand_3x3_conv_'+name, out_squeeze, out_channel=num_expand_3x3_filters, kernel_shape=3, stride=1, padding='SAME')
    out_expand_3X3 = tf.nn.relu(out_expand_3X3)
    if skip is 0:
        return out_expand_3X3
    else:
        return tf.add(x, out_expand_3X3)


def monitor(x, name):
    return tf.Print(x, [x], message='\n\n' + name + ': ', summarize=1000, name=name)


def GenerateGrid(grid_size):
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)# the grid shape becomes (gridSize,gridSize,1,2)
    grid = tf.tile(grid,tf.constant([1,1,3,1], tf.int32) )#the grid shape becomes (gridSize,gridSize,3,2)
    grid=tf.cast(grid,tf.float32)
    return grid


def compute_iou(boxes1, boxes2):
    boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                         boxes1[..., 1] - boxes1[..., 3] / 2.0,
                         boxes1[..., 0] + boxes1[..., 2] / 2.0,
                         boxes1[..., 1] + boxes1[..., 3] / 2.0],
                        axis=-1)

    boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                         boxes2[..., 1] - boxes2[..., 3] / 2.0,
                         boxes2[..., 0] + boxes2[..., 2] / 2.0,
                         boxes2[..., 1] + boxes2[..., 3] / 2.0],
                        axis=-1)
    lu = tf.maximum(tf.cast(boxes1_t[..., :2],dtype=tf.float32), tf.cast(boxes2_t[..., :2],dtype=tf.float32))
    rd = tf.minimum(tf.cast(boxes1_t[..., 2:],dtype=tf.float32), tf.cast(boxes2_t[..., 2:],dtype=tf.float32))

    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

    square1 = boxes1[..., 2] * boxes1[..., 3]
    square2 = boxes2[..., 2] * boxes2[..., 3]

    union_square = tf.maximum(tf.cast(square1,dtype=tf.float32) + tf.cast(square2,dtype=tf.float32) - tf.cast(inter_square,dtype=tf.float32), 1e-10)
    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


class PolypDetectionModel:
    def __init__(self, data_path=None, training=True):
        self.data_path = data_path
        self.dataset = polyp_dataset(data_path=self.data_path)
        self.training = training
        self.n_grid = 4
        self.n_boxes = 3

    def get_dataset(self):
        return self.dataset.load_train_data()

    def first_layer(self, x):
        l = Conv2D("First", x, out_channel=16, kernel_shape=3, stride=1, padding="SAME")
        return l

    def reshape_output(self, outputs):
        grid = GenerateGrid(4)
        objectness_net_out, pred_box_offset_coord = tf.split(outputs, (1, 4), axis=-1)

        pred_box_normalized_coord_CxCy = (pred_box_offset_coord[:, :, :, :, 0:2] + grid) / 4
        pred_box_normalized_coord_wh = tf.square(pred_box_offset_coord[:, :, :, :, 2:])

        box_x1y1 = pred_box_normalized_coord_CxCy - pred_box_normalized_coord_wh / 2
        box_x2y2 = pred_box_normalized_coord_CxCy + pred_box_normalized_coord_wh / 2
        box_x1y1x2y2_withLUAs0_scale01 = tf.concat([box_x1y1, box_x2y2], axis=-1)

        return box_x1y1x2y2_withLUAs0_scale01, objectness_net_out

    def reshape_output_for_prediction(self, outputs):
        b, c = [], []
        o = outputs
        num_tot_boxes = self.n_grid*self.n_grid*self.n_boxes
        bbox = tf.reshape(o[0], (1, num_tot_boxes, 1, 4))
        confidence = tf.reshape(o[1], (1, num_tot_boxes, 1))
        scores = confidence
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=bbox,
            scores=confidence,
            max_output_size_per_class=100,
            max_total_size=100,
            iou_threshold=0.5,
            score_threshold=0.5
        )
        return boxes, scores, classes, valid_detections

    def get_model(self, X, training=True):
        l = self.first_layer(X)
        l = tf.nn.relu(l)
        l = MaxPooling('pool1', l, shape=3, stride=2, padding='SAME')
        l = halffire('hafflire1', l)
        l = MaxPooling('pool2', l, shape=3, stride=2, padding='SAME')
        l = halffire('hafflire2', l)
        l = MaxPooling('pool3', l, shape=3, stride=2, padding='SAME')
        l = halffire('hafflire3', l)
        l = MaxPooling('pool4', l, shape=3, stride=2, padding='SAME')
        l = halffire('hafflire4', l)
        l = MaxPooling('pool5', l, shape=3, stride=2, padding='SAME')
        l = halffire('hafflire5', l)
        l = MaxPooling('pool5', l, shape=3, stride=2, padding='SAME')
        l = halffire('hafflire6', l)
        l = halffire('hafflire7', l)
        output_0 = Conv2D('output', l, out_channel=3 * 5, kernel_shape=1, stride=1, padding='SAME')
        output_0 = tf.reshape(output_0, shape=(-1, 4, 4, 3, 5), name="FinalOutput")
        if training is False:
            output_0 = self.reshape_output(output_0)
            output_0 = self.reshape_output_for_prediction(output_0)
            return output_0
        return output_0

    def get_loss(self, y_true=None, y_pred=None, train_state=True, grid_size=4, n_boxes=3):
        grid = GenerateGrid(grid_size=4)
        pred_obj_conf = y_pred[:, :, :, :, 0]
        pred_box_offset_coord = y_pred[:, :, :, :, 1:]

        pred_box_normalized_coord = tf.concat([(pred_box_offset_coord[:, :, :, :, 0:2] + grid) / grid_size,
                                               tf.square(pred_box_offset_coord[:, :, :, :, 2:])], axis=-1)

        target_obj_conf = y_true[:, :, :, 0]
        target_obj_conf = tf.reshape(target_obj_conf, shape=[-1, grid_size, grid_size, 1])
        target_obj_conf = tf.cast(target_obj_conf, dtype=tf.float32)

        target_box_coord = y_true[:, :, :, 1:]
        target_box_coord = tf.reshape(target_box_coord, shape=[-1, grid_size, grid_size, 1, 4])
        target_box_coord_aT = tf.tile(target_box_coord, multiples=[1, 1, 1, n_boxes, 1])
        target_box_normalized_coord = target_box_coord_aT

        target_box_offset_coord = tf.concat(
            [tf.cast(target_box_normalized_coord[:, :, :, :, 0:2] * grid_size, dtype=tf.float32) - grid,
             tf.cast(tf.sqrt(target_box_normalized_coord[:, :, :, :, 2:]), dtype=tf.float32), ], axis=-1)

        pred_ious = compute_iou(target_box_normalized_coord, pred_box_normalized_coord)
        predictor_mask_max = tf.reduce_max(pred_ious, axis=-1, keepdims=True)
        predictor_mask = tf.cast(pred_ious >= tf.cast(predictor_mask_max, dtype=tf.float32),
                                 tf.float32) * target_obj_conf
        noobj_mask = tf.ones_like(predictor_mask) - predictor_mask

        # computing the confidence loss
        obj_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(predictor_mask * (pred_obj_conf - predictor_mask)), axis=[1, 2, 3]))
        noobj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobj_mask * pred_obj_conf), axis=[1, 2, 3]))

        # computing the localization loss
        predictor_mask_none = predictor_mask[:, :, :, :, None]
        loc_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(predictor_mask_none * (target_box_offset_coord - pred_box_offset_coord)),
                          axis=[1, 2, 3]))

        loss = 10 * loc_loss + 2 * obj_loss + 0.5 * noobj_loss

        if train_state is True:
            tf.summary.scalar("loc_loss", K.sum(10 * loc_loss))
            tf.summary.scalar("obj_loss", K.sum(2 * obj_loss))
            tf.summary.scalar("nonObj_loss", K.sum(0.5 * noobj_loss))
            # print("loss is:{}".format(loss))
        return loss