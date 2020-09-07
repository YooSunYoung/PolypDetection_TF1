from absl import app, flags
from absl.flags import FLAGS
import random
import cv2
import os
flags.DEFINE_string("video_path", "../data/WithoutPolypVideo/OUH_10050448_08_06_15_1.mpg", "path to the video file")
flags.DEFINE_string("output_dir", "../data/NoPolypImages/", "directory for output images")
flags.DEFINE_integer("output_image_num", 3, "number of output images you want to sample from the video")


def extract_image(video_path, output_image_num, output_dir):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)

    video_len = video.get(cv2.CAP_PROP_POS_MSEC)
    video_frame_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    temp_num = 0
    prefix = output_dir if not output_dir.endswith("/") else output_dir[:-1]
    while os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        output_dir = prefix + "_{}".format(temp_num)
        temp_num += 1
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    frame_indices = random.sample(range(1, video_frame_len+1), output_image_num)

    for idx_frame in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx_frame-1)  # this code set the offset as idx_frame - 1
        ret, frame = video.read()
        cv2.imwrite(os.path.join(output_dir, "no_polyp_{}.jpg".format(idx_frame)), frame)

    video.release()
#    cv2.destroyAllWindows()


def main(_args):
    video_path = FLAGS.video_path
    output_image_num = FLAGS.output_image_num
    output_dir = FLAGS.output_dir
    extract_image(video_path, output_image_num, output_dir)
    return 0


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
