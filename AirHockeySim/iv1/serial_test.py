from serial import Serial
import time

class serial_connect():
    def __init__(self, com='com3', baud=115200):
        self.connection = Serial(com, baud)

    def send(self, x, y):
        data = ('<'+ '%03d' % y + '%03d' % (x+260) + '>').encode()
        self.connection.write(data)
        time.sleep(0.005)

    def __del__(self):
        self.connection.close()



if __name__ == "__main__":
    a_con = serial_connect()

    while True:
        x = int(input("x: "))
        y = int(input("y: "))
        print("sending [x: " + str(x) + ", y: " + str(y) + "]\n" + str(y) + " " + str((x+260)))
        a_con.send(x, y)