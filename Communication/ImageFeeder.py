from Communication.SocketCommunicator import SocketCommunicator
import os
import cv2


class ImageFeeder(SocketCommunicator):
    def __init__(self, **kwargs):
        kwargs['server'] = False
        SocketCommunicator.__init__(self, **kwargs)
        self.image_directory = kwargs.get("directory", "../data/NoPolypImages/")
        self.image_paths = None
        self.sent_images = dict({})
        self.received_results = []
        self.ready = False

    def build_connection(self):
        SocketCommunicator.build_connection(self, server=False)

    def get_ready(self):
        self.send_string("ready")
        rd = self.receive_string(5)
        if rd == 'ready':
            self.ready = True
        else:
            self.ready = False

    def scrape_image_file_paths(self):
        image_files = os.listdir(self.image_directory)
        image_file_names = [x for x in image_files if x.endswith(".jpg")]
        self.image_paths = [os.path.join(self.image_directory, file) for file in image_file_names]
        return self.image_paths

    def send_single_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.debug_mode: print(image[:3])
        self.send_array(image, len(image))

    def send_images(self):
        if self.connection is None:
            print("Connection not established.")
            return
        self.scrape_image_file_paths()
        self.get_ready()
        if self.ready:
            self.send_integer(len(self.image_paths))
            while len(self.image_paths) > 0:
                image_path = self.image_paths.pop(0)
                self.send_single_image(image_path)
                self.sent_images[image_path] = None
        else:
            print("Communication should start with 'ready'.")

    def receive_results(self):
        for key in self.sent_images.keys():
            self.sent_images[key] = self.receive_array(5)
        print(self.sent_images)
        return self.sent_images


if __name__ == "__main__":
    image_feeder = ImageFeeder(port=6007,
                               server_port=6003,
                               debug_mode=False,
                               directory="../data/TimeConsumption/1000/")
    image_feeder.build_connection()
    image_feeder.send_images()
    image_feeder.receive_results()
    image_feeder.close_connection()
