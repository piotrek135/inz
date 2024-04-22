import cv2, time, pandas
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

data_for_graph = []
img_num = 0
path = 'F:\\zzzz STUDIA\\sem6\\projekt dyplomowy  inz\\motion_detection_for_capturing_bird_pics\\'

take_pics = [0,0]

vid = cv2.VideoCapture(0)

ret, prev_pic = vid.read()
prev_pic = cv2.resize(prev_pic, (640,360))

def save_sharpest():
    images = []
    gray = []
    gradient_x = []
    gradient_y = []
    sharpness = []
    for i in range(5):
        images.append(cv2.imread(os.path.join(path ,f'{take_pics[1]}\\{img_num-5+i}.jpg')))
        gray.append(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY))
        gradient_x.append(cv2.Sobel(gray[i], cv2.CV_64F, 1, 0, ksize=3))
        gradient_y.append(cv2.Sobel(gray[i], cv2.CV_64F, 0, 1, ksize=3))
        sharpness.append(np.sqrt(gradient_x[i]**2 + gradient_y[i]**2).mean())

    max_sharpness_index = 0
    max_sharpness = sharpness[0]
    for i in range(5):
        if max_sharpness<sharpness[i]:
            max_sharpness = sharpness[i]
            max_sharpness_index = i

    cv2.imwrite(os.path.join(path , f'sharp\\{img_num-5+max_sharpness_index}.jpg'), images[max_sharpness_index])
    return


while(True):
    ret, frame = vid.read()
    frame = cv2.resize(frame, (640,360))
    cv2.imshow('1', frame)
    diff = cv2.absdiff(prev_pic, frame)
    cv2.imshow('2', diff)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    cv2.imshow('3', gray_diff)
    thresh_diff = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('4', thresh_diff)

    total_pixels = prev_pic.shape[0] * prev_pic.shape[1] * 1.0
    diff_on_pixels = cv2.countNonZero(thresh_diff) * 1.0
    difference_measure = diff_on_pixels / total_pixels

    data_for_graph.append(difference_measure)

    if take_pics[0] > 0:
        print(take_pics)
        cv2.imwrite(os.path.join(path , f'{take_pics[1]}\\{img_num}.jpg'), frame)
        take_pics[0] -= 1
        img_num += 1
        if take_pics[0] == 0:
            time.sleep(1)
            ret, frame = vid.read()
            frame = cv2.resize(frame, (640,360))
            save_sharpest()
    else:
        take_pics[0] = 4
        img_num += 1
        if (difference_measure > .2):
            take_pics[1] = 20
        elif (difference_measure > .15):
            take_pics[1] = 15
        elif (difference_measure > .10):
            take_pics[1] = 10
        elif (difference_measure > .05):
            take_pics[1] = 5
    prev_pic = frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        x = range(len(data_for_graph))
        y = data_for_graph
        plt.plot(x, y)
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.show()

        with open('plik.txt', 'w') as f:
            for line in data_for_graph:
                f.write(f"{line}\n")
        break
    time.sleep(0.2)

vid.release()
cv2.destroyAllWindows()