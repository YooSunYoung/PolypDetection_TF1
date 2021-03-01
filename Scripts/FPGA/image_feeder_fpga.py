from Communication.ImageFeeder import ImageFeeder

if __name__ == "__main__":
    image_feeder = ImageFeeder( ip_address='192.168.10.123',
                                server_ip='192.168.10.125',
                               port=6007,
                               server_port=6004,
                               debug_mode=True,
                               directory="C:\\Users\syo\PycharmProjects\PolypDetection_TF1\data\TimeConsumption\journal")
    image_feeder.build_connection()
    image_feeder.send_images()
    image_feeder.receive_results()
    image_feeder.close_connection()
