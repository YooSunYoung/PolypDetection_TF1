import socket
import numpy as np


class SocketCommunicator:
    def __init__(self, **kwargs):
        self.debug_mode = kwargs.get('debug_mode', True)
        self.ip_address = kwargs.get('ip_address', 'localhost')
        self.port = kwargs.get('port', 6006)
        self.server = kwargs.get('server', True)
        if not self.server:
            self.server_port = kwargs.get('server_port', 6006)
        self.connection = None

    def build_server_connection(self):
        address = (self.ip_address, self.port)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(address)
        s.listen(5)
        conn, addr = s.accept()
        if self.debug_mode:
            print('[+] Connected with ', addr)
        self.connection = conn
        return conn

    def build_client_connection(self):
        while True:
            try:
                self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                address = (self.ip_address, self.port)
                self.connection.bind(address)
                self.connection.connect(('localhost', self.server_port))
            except OSError as e:
                if e.args[0] == 10048:
                    print("Port number {} is not available.".format(self.port))
                    self.port += 1
                    print("Reconnecting with port number {}.".format(self.port))
                else:
                    self.connection = None
                    break
            else:
                break
        return self.connection

    def build_connection(self, server=True):
        if server:
            return self.build_server_connection()
        else:
            return self.build_client_connection()

    def close_connection(self):
        self.connection.close()

    def receive_string(self, len_string):
        data = self.connection.recv(len_string)
        data = data.decode("UTF-8")
        return data

    def receive_integer(self, len_integer):
        data = self.connection.recv(len_integer)
        data = data.decode("UTF-8")
        data = int(data)
        return data

    def receive_array(self, len_array):
        len_received_data = 0
        received_data = []
        while len_received_data < len_array:
            data = self.connection.recv(len_array)
            received_data.extend(data)
            len_received_data = len_received_data + data.__len__()
            if self.debug_mode:
                print("Received data length = {}\n".format(len_received_data))
            if not data:
                break
        return received_data

    def send_string(self, line, length=5):
        self.connection.send(line.zfill(length).encode("UTF-8"))

    def send_integer(self, number, length=5):
        self.connection.send(str(number).zfill(length).encode("UTF-8"))

    def send_array(self, array, length=5):
        if type(array) is not np.ndarray:
            array = np.array(array)
        self.connection.send(array)
