import cv2
import numpy as np
import time

def get_m():
    with open('.\\calib_save.txt', 'r') as c:
        lines = c.readlines()

    for i in range(len(lines)):
        lines[i] = float(lines[i].replace('\n', ''))
    M = np.array([[lines[0], lines[1], lines[2]], [lines[3], lines[4], lines[5]], [lines[6], lines[7], lines[8]]])
    maxWidth = int(lines[9])
    maxHeight = int(lines[10])

    return M, maxWidth, maxHeight

def four_point_transform(image, M, maxWidth, maxHeight):
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def filter_green(image):
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #lower_green = np.array([38, 100, 100])
    #upper_green = np.array([75, 255, 255])
    lower_green = np.array([29, 86, 6])
    upper_green = np.array([64, 255, 255])

    mask_g = cv2.inRange(image_hsv, lower_green, upper_green)

    output_img_g = image.copy()
    output_img_g[np.where(mask_g == 0)] = 0

    return output_img_g

def main():
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    frame_times = []
    M, maxWidth, maxHeight = get_m()
    while(True):
        start = time.time()

        ret, frame = vid.read()

        warped_o = four_point_transform(frame, M, maxWidth, maxHeight)
        output_img_g = filter_green(warped_o)       
        cv2.imshow('output_img_g', output_img_g)
        g_gry_img = cv2.cvtColor(output_img_g, cv2.COLOR_BGR2GRAY)

        green_circles = cv2.HoughCircles(g_gry_img,cv2.HOUGH_GRADIENT,1,20,param1=100,param2=8,minRadius=0,maxRadius=30)
        g_pts = np.array([])
        if green_circles is not None:
            green_circles = np.round(green_circles[0, :]).astype("int")
            num = len(green_circles)
            if num:
                xs = 0
                ys = 0
                rs = 20
                for (x, y, r) in green_circles:
                    xs += x
                    ys += y
                    g_pts = np.array([xs/num, ys/num, rs])
            else:
                g_pts = np.array([x, y, r])
            
                if g_pts.size == 0:
                    g_pts = np.array([x, y])
                else:
                    g_pts=np.append(g_pts, [[x, y]], axis = 0)          

        end = time.time()
        frame_times.append(end-start)
        if len(frame_times) > 100:
            del frame_times[0]
        avg_time = sum(frame_times)/len(frame_times)

        cv2.putText(warped_o, str(int(1/avg_time)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2, lineType=cv2.LINE_AA)
        
        if green_circles is not None:
            print(g_pts)
            cv2.circle(warped_o, (int(g_pts[0]), int(g_pts[1])), (int(g_pts[2])), (0, 255, 0), -1)

        cv2.imshow('warped_o', warped_o)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    vid.release() 
    cv2.destroyAllWindows() 

if __name__=='__main__':
    main()