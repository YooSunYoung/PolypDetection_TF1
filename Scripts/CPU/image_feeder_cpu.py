from Communication.ImageFeeder import ImageFeeder
import sys

if __name__ == "__main__":
    image_directory = sys.argv[1]
    image_feeder = ImageFeeder(port=6007,
                               server_port=6004,
                               debug_mode=False,
                               directory=image_directory)
                                # directory="../../data/TimeConsumption/2/")
    image_feeder.build_connection()
    image_feeder.send_images()
    image_feeder.receive_results()
    for key in image_feeder.sent_images.keys():
        print(image_feeder.sent_images[key])
    image_feeder.close_connection()
