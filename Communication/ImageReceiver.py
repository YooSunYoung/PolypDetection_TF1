from Communication.SocketCommunicator import SocketCommunicator
import numpy as np


class ImageReceiver(SocketCommunicator):
    def __init__(self, **kwargs):
        SocketCommunicator.__init__(self, **kwargs)
        self.life_time = kwargs.get('life_time', 1)
        self.data_queue = []
        self.ready = False

    def build_connection(self):
        SocketCommunicator.build_connection(self, server=True)

    def get_ready(self):
        rd = self.receive_string(5)
        if rd == 'ready':
            self.ready = True
        else:
            self.ready = False
        self.send_string("ready")

    def receive_images(self):
        if self.connection is None:
            print("Connection not established.")
            return
        self.get_ready()
        imgs = []
        if self.ready:
            num_image = self.receive_integer(5)
            for i in range(num_image):
                img = self.receive_array(154587)
                imgs.append(img)
        else:
            print("Communication should start with 'ready'.")
        return imgs


def dummy_analysis(data):
    return [0,0,0,0,0]


if __name__ == "__main__":
    receiver = ImageReceiver(debug_mode=False)
    receiver.build_connection()
    images = receiver.receive_images()
    results = []
    for image in images:
        results.append(dummy_analysis(image))
    print(results)
    for result in results:
        receiver.send_array(result)
    receiver.close_connection()
