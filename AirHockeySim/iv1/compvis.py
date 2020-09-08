import cv2
import numpy as np
import time

class puck_detect():
    def __init__(self):
        self.transformation_matrix = None
        self.maxWidth = 0
        self.maxWidth = 0
        self.vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.get_calibration_metrics()

        # DATA
        self.frame = None
        self.g_pts = np.array([])

        # PARAMTERS
        #self.lower_green = np.array([29, 86, 6])
        #self.upper_green = np.array([64, 255, 255])
        #self.lower_green = np.array([30, 86, 8])
        #self.upper_green = np.array([60, 255, 255])
        self.lower_green = np.array([30, 90, 10])
        self.upper_green = np.array([60, 235, 235])
        self.dp = 1
        self.min_dist = 20
        self.param1 = 100
        self.param2 = 7 #8 
        self.min_radius = 10
        self.max_radius = 40

        # DEBUG
        self.debug = True
        if self.debug:
            self.o_frame = None
            self.w_frame = None
            self.g_frame = None
            self.p_frame = None
            self.frame_times = []

    def get_calibration_metrics(self):
        with open('.\\calib_save.txt', 'r') as c:
            lines = c.readlines()

        for i in range(len(lines)):
            lines[i] = float(lines[i].replace('\n', ''))
        self.transformation_matrix = np.array([[lines[0], lines[1], lines[2]],\
                    [lines[3], lines[4], lines[5]], [lines[6], lines[7], lines[8]]])

        self.maxWidth = int(lines[9])
        self.maxHeight = int(lines[10])

    def four_point_transform(self):
        if self.debug:
            self.o_frame = self.frame.copy()
        self.frame = cv2.warpPerspective(self.frame, self.transformation_matrix,\
                                        (self.maxWidth, self.maxHeight))
        
        self.frame = cv2.resize(self.frame, (1100, 550))
        if self.debug:
            self.w_frame = self.frame.copy()

    def filter_green(self):
        image_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask_g = cv2.inRange(image_hsv, self.lower_green, self.upper_green)
        #self.frame[np.where(mask_g == 0)] = 0
        self.frame = cv2.bitwise_and(self.frame, self.frame,mask=mask_g)
        if self.debug:
            self.g_frame = self.frame
        
    def get_puck_location(self):
        if self.debug:
            start = time.time()
        _, self.frame = self.vid.read()
        self.four_point_transform()
        self.filter_green()
        g_gry_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        green_circles = cv2.HoughCircles(g_gry_img,cv2.HOUGH_GRADIENT,self.dp,self.min_dist,\
                                            param1=self.param1,param2=self.param2,\
                                            minRadius=self.min_radius,maxRadius=self.max_radius)    
        
        if green_circles is not None:
            green_circles = np.round(green_circles[0, :]).astype("int")
            num = len(green_circles)
            if num > 1:
                xs = 0
                ys = 0
                rs = 20
                for (x, y, r) in green_circles:
                    xs += x
                    ys += y
                    self.g_pts = np.array([xs/num, ys/num, rs])
            else:
                self.g_pts = np.array([green_circles[0][0], green_circles[0][1], 20])    

        if self.debug:
            self.p_frame = self.w_frame.copy()
            end = time.time()
            self.frame_times.append(end-start)
            if len(self.frame_times) > 100:
                del self.frame_times[0]
            avg_time = sum(self.frame_times)/len(self.frame_times)
            cv2.putText(self.p_frame, str(int(1/avg_time)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,\
                    1, (0, 200, 255), 2, lineType=cv2.LINE_AA)

            if green_circles is not None:
                cv2.circle(self.p_frame, (int(self.g_pts[0]), int(self.g_pts[1])), (int(self.g_pts[2])),\
                            (0, 255, 0), -1)
            
            cv2.imshow('processed frame', self.p_frame)
            cv2.imshow('green frame', self.g_frame)
            cv2.waitKey(1)

        return self.g_pts

    def get_p_frame(self):
        return self.p_frame.copy()

    def __del__(self):
        self.vid.release() 
        cv2.destroyAllWindows() 