from Communication.ImageFeeder import ImageFeeder

if __name__ == "__main__":
    image_feeder = ImageFeeder(port=6007,
                               server_port=6004,
                               debug_mode=True,
                               directory="../../data/TimeConsumption/1000/")
    image_feeder.build_connection()
    image_feeder.send_images()
    image_feeder.receive_results()
    image_feeder.close_connection()
