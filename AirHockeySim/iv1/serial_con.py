from serial import Serial
import time

class serial_connect():
    def __init__(self, com='com4', baud=115200):
        self.connection = Serial(com, baud)

    def send(self, x, y):
        data = ('<'+ '%03d' % (x) + '%03d' % (y+260) + '>').encode()
        self.connection.write(data)
        #time.sleep(0.005)

    def __del__(self):
        self.connection.close()