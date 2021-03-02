configuration = {
    "image_width": 227,
    "image_height": 227,
    "grid_size": 4,
    "n_boxes": 3,
    "n_channels": 3,
    "n_classes": 0,
    "model_name": "squeeze_tiny"
}


class GlobalVar:
    num_conv = 0
    save_conv_input_output = False
    score_threshold = 0.0


def get_num_conv():
    return GlobalVar.num_conv


def set_num_conv(num):
    GlobalVar.num_conv = num


def get_save_conv_input_output():
    return GlobalVar.save_conv_input_output


def get_score_threshold():
    return GlobalVar.score_threshold


def set_score_threshold(num):
    GlobalVar.score_threshold = num
