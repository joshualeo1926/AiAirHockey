import cv2
import numpy as np
import time

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    buff = 150
    height_buffer = 150
    width_buffer = 130
    top = 0
    right = 20
    left = 20
    bot = 20
    rect[0] = rect[0] - [height_buffer+0, width_buffer-5]
    rect[1] = rect[1] + [height_buffer+0, -width_buffer+15]
    rect[2] = rect[2] + [height_buffer, width_buffer-10]
    rect[3] = rect[3] + [-height_buffer, width_buffer-15]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    with open('.\\calib.txt', 'w') as c:
        matrix = ''
        for l in M:
            for v in l:
                matrix += str(v) + '\n'

        c.write(matrix + str(maxWidth) + '\n' + str(maxHeight) + '\n')
        print(matrix + str(maxWidth) + '\n' + str(maxHeight) + '\n')

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def filter_red(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask_r_0 = cv2.inRange(image_hsv, lower_red, upper_red)
    lower_red = np.array([160, 50, 50])
    upper_red = np.array([179, 255, 255])
    mask_r_1 = cv2.inRange(image_hsv, lower_red, upper_red)
    mask_r = mask_r_0 + mask_r_1

    output_img_r = image.copy()
    output_img_r[np.where(mask_r==0)] = 0

    output_hsv_r = image_hsv.copy()
    output_hsv_r[np.where(mask_r == 0)] = 0

    return output_img_r, output_hsv_r

def main():
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 550)
    frame_times = []

    while(True):
        start = time.time()

        ret, frame = vid.read()
        print(frame)
        output_img_r, output_hsv_r = filter_red(frame)

        r_blur = cv2.blur(output_img_r,(3,3))
        r_gry_img = cv2.cvtColor(r_blur, cv2.COLOR_BGR2GRAY)

        r_gry_img[np.where(r_gry_img<40)] = 0
        r_gry_img[np.where(r_gry_img>40)] = 255

        red_circles = cv2.HoughCircles(r_gry_img,cv2.HOUGH_GRADIENT,1,150,param1=200,\
                                        param2=10,minRadius=60,maxRadius=75)
               
        r_pts = np.array([])
        if red_circles is not None:
            red_circles = np.round(red_circles[0, :]).astype("int")
            for (x, y, r) in red_circles:
                if r_pts.size == 0:
                    r_pts = np.array([[x, y]])
                else:
                    r_pts=np.append(r_pts, [[x, y]], axis = 0)        

        if r_pts.size > 0:
            warped_o = four_point_transform(frame, r_pts)

        end = time.time()
        frame_times.append(end-start)
        if len(frame_times) > 100:
            del frame_times[0]
        avg_time = sum(frame_times)/len(frame_times)

        cv2.putText(frame, str(int(1/avg_time)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,\
                    (0, 200, 255), 2, lineType=cv2.LINE_AA)

        if red_circles is not None:
            for circle in red_circles:
                cv2.circle(frame, (int(circle[0]), int(circle[1])), (int(circle[2])),\
                            (0, 0, 255), 2)

        cv2.imshow('original', frame)
        if r_pts.size > 0:
            cv2.imshow('warped_o', warped_o)
        cv2.imshow('red', output_img_r)

        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('k')]:
            key = cv2.waitKey(0)
        if key == ord('q'):
            break

    vid.release() 
    cv2.destroyAllWindows() 

if __name__=='__main__':
    main()