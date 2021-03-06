import numpy as np
import math
import cv2
import time 
import sys, os, os.path
from TD3Agent import TDAgent
from compvis import puck_detect
from serial_con import serial_connect
import matplotlib.pyplot as plt

running = True

class puck_stats():
    def __init__(self):
        self.last_x = 0
        self.last_y = 0
        self.current_x = 0
        self.current_y = 0
        self.x_vel = 0
        self.y_vel = 0
        self.last_x_vel = 0
        self.last_y_vel = 0
        
        self.threshold_vel = 60
        self.friction = 0.989
        self.first_detect = True

    def update(self, pos, dt):
        if (((pos[0]-self.last_x)/dt)**2 + ((pos[1]-self.last_y)/dt)**2)**0.5 < self.threshold_vel and False:
            self.current_x = self.last_x
            self.current_y = self.last_y
        else:
            self.current_x = pos[0]
            self.current_y = pos[1]
        self.x_vel = (self.current_x-self.last_x)/dt
        self.y_vel = (self.current_y-self.last_y)/dt    
        self.last_x = self.current_x
        self.last_y = self.current_y
        self.last_x_vel = self.x_vel
        self.last_y_vel = self.y_vel
        return self.current_x, self.x_vel, self.current_y, self.y_vel
    
class paddle_stats():
    def __init__(self):
        self.x = 56
        self.y = 250
        self.last_x = 0
        self.last_y = 0
        self.current_x = 0
        self.current_y = 0
        self.x_vel = 0
        self.y_vel = 0
        self.x_accel = 0
        self.y_accel = 0
        self.velocity = 0

        self.max_vel = 500
        self.max_accel = 6000

        # LIMITS
        self.x_max = 360
        self.x_min = 0
        self.y_max = 520
        self.y_min = 0

    def update(self, choice, time_delta):
        x_accel = self.max_accel * choice[0]
        y_accel = self.max_accel * choice[1]

        self.x_vel += x_accel * time_delta
        self.y_vel += y_accel * time_delta
        if math.sqrt(self.x_vel**2 + self.y_vel**2) > self.max_vel:
            self.x_vel = choice[0] * self.max_vel
            self.y_vel = choice[1] * self.max_vel

        mult = 1 # 1.1
        self.x += mult * self.x_vel * time_delta
        self.y -= mult * self.y_vel * time_delta

        if self.x > self.x_max:
            self.x = self.x_max
            
        if self.x < self.x_min:
            self.x = self.x_min

        if self.y > self.y_max:
            self.y = self.y_max

        if self.y < self.y_min:
            self.y = self.y_min


        self.x_vel = (self.x - self.last_x)/time_delta
        self.y_vel = (self.last_y - self.y)/time_delta
        self.velocity = math.sqrt(self.x_vel**2 + self.y_vel**2)

        self.last_x = self.x
        self.last_y = self.y

        return self.x, self.x_vel, self.y, self.y_vel

if __name__ == "__main__":
    debug = True

    # CONNECT TO ARDUINO SERIAL
    arduino_link = serial_connect()

    # INITALISE TD3 ML AGENT
    agent1 = TDAgent(alpha=5e-5, beta=5e-5, input_dims=[8], tau=0.005, max_action=1.0, 
                    min_action=-1.0, gamma=0.99, update_actor_interval=20, 
                    warmup=0, n_actions=2, max_size=1000000, 
                    layer1_size=400, layer2_size=300, batch_size=100,
                    noise=0) #noise=0.1

    # LOAD TRAINED STATE DICT
    agent1.load_models()

    # INITALISE COMPUTER VISION DETECTOR
    detector = puck_detect(debug)

    # INIALISE PUCK AND PADDLE LOGGERS
    puck_stat = puck_stats()
    paddle_stat = paddle_stats()

    point = detector.get_puck_location()
    # STATE VECTOR
    state = np.array([0.0, 0.0, 275.0, 0.0, point[0], 0.0, point[1], 0.0])

    last_time = time.time()
    while running:
        
        # GET PUCK LOCATION
        point = detector.get_puck_location()

        # GET TIME DIFFERENCE
        current_time = time.time()
        dt = current_time - last_time
        
        # UPDATE PUCK LOCATION
        puck_x, puck_x_vel, puck_y, puck_y_vel = puck_stat.update(point, dt)

        # UPDATE STATE VECTOR
        state[4:8] = [puck_x, puck_x_vel, puck_y, puck_y_vel]

        # CHOSE AN ACTION
        action = agent1.choose_actions(state)
        # UPDATE PADDLE LOCATION
        paddle_x, paddle_x_vel, paddle_y, paddle_y_vel = paddle_stat.update(action, dt)

        # EXECUTE ACTION
        if paddle_y+-20 > 520:
            val1 = 520
        else:
            val1 = paddle_y+-20
        
        if paddle_x-0 < 30:
            val2 = 30
        else:
            val2 =  paddle_x-0

        arduino_link.send(int(val1), int(val2))

        # UPDATE STATE VECTOR
        state[0:4] = [paddle_x, paddle_x_vel, paddle_y, paddle_y_vel]

        # UPDATE TIME 
        last_time = time.time()

        # DEBUG
        if debug == True:
            frame = detector.get_p_frame()
            cv2.circle(frame, (int(paddle_x), int(paddle_y)), (int(20)), (0, 255, 255), -1)
            cv2.circle(frame, (int(puck_x), int(puck_y)), (int(30)), (255, 0, 0), 5)

            cv2.putText(frame, "x: % 3d" % int(abs(paddle_x)) + " y: % 3d" % int(abs(paddle_y)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, "x: % 3d" % int(abs(puck_x)) + " y: % 3d" % int(abs(puck_y)), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2, lineType=cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)