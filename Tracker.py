import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from Draw_Polygons import PolygonDrawer
import pandas as pd 
import math
import sys
import os
from tqdm import tqdm

if len(sys.argv)>1:
    video_name = sys.argv[1]
else:
    print("Please enter video path!!!")

if len(sys.argv)>2:
    exp_type = sys.argv[2]
else:
    print("Please enter experiment type!!!")

if len(sys.argv)>3:
    actual_area = sys.argv[3]
else:
    print("Please enter experiment type!!!")
     

# ***** replace with required image path *****
cap = cv2.VideoCapture(video_name)
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( total_frame )
# ***** global variable decleration *****
_, frame1 = cap.read()
_, frame2 = cap.read()

# Back Graund Substraction
backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False)

#roi = (239, 200, 117, 114)

## Polygon Drawer Class
pd_ = PolygonDrawer("Polygon",frame1)
image = pd_.run(frame1)

#fps for calculate time
fps = cap.get(5)
pts = []
print(pd_.points)


# for plus maze there is 4 polygon to separate experimant area
if exp_type == 'plusmaze':

    polyRight = Polygon(pd_.points[0])

    polyleft = Polygon(pd_.points[1])

    polyUp = Polygon(pd_.points[2])

    polyDown = Polygon(pd_.points[3])

# for plus maze there is 2 polygon to separate experimant area

if exp_type == 'openfield':
    pd_.points =   [[(620, 440), (621, 502), (965, 511), (966, 454)], [(540, 431), (539, 506), (218, 492), (222, 436)], [(564, 65), (634, 64), (621, 437), (541, 430)], [(540, 506), (619, 503), (609, 884), (534, 885)]]
                  
    polyOutside  = Polygon(pd_.points[0])

    polyInside = Polygon(pd_.points[1])

#polyMid = Polygon(pd.points[4])
framecounts = []
framecount = 0
flag = 0
times = []
total_path=0
count = 0

total_path=0
right_= 0 
left_= 0
up_ = 0
down_ = 0
mid_ = 0

r_x = 0
l_x = 999
if exp_type == 'plusmaze':
    for i in pd_.points[0]:
        x,y = i 
        if x > r_x:
            r_x = x

    for i in pd_.points[1]:
        x,y = i 
        if x < l_x:
            l_x = x

if exp_type == 'openfield':
    for i in pd_.points[0]:
        x,y = i 
        if x > r_x:
            r_x = x
        if x < l_x:
            l_x = x
    
        
flag = 0
inside = 0 
outside = 0
dist_exp_area = abs(r_x - l_x)
counter= 0

pbar = tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),desc="Creating output..")

while cap.isOpened():
    if counter == total_frame -1 :
        break
    # Here is tracking methods  with order : backgraund subsraction - threshold - dilate - find contours
    mask = backSub.apply(frame1)
    #diff = cv2.absdiff(frame1, frame2)
    _,thresh = cv2.threshold(mask,20,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh,None,iterations=3)
    contours,_ = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # here find biggest contour and draw rectangle around it after find center point and check if that point in polygons which we describe or not and count how many second there
    for contour in contours:
     
        #(x,y,w,h) = cv2.boundingRect(contour)


        if cv2.contourArea(contour) <700:
            continue
        M = cv2.moments(contour)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        
        
        center = (cx, cy)
        a = Point(center)
        
        if exp_type == 'plusmaze':
            place = []
            
            if counter > 60:
                pts.append(center)
            p = Point((cx, cy))
            if p.within(polyRight):
                right_ += 1
                if len(place) <1:
                    place.append("Right Side")
            elif p.within(polyleft):
                left_ += 1
                if len(place) <1:
                    place.append("Left Side")
            elif p.within(polyUp):
                up_ += 1
                if len(place) <1:
                    place.append("Up Side")
            elif a.within(polyDown):
                down_ += 1
                if len(place) <1:
                    place.append("Down Side")
            else:
                mid_ += 1
        if exp_type == 'openfield':
            p = Point((cx, cy))

            if p.within(polyInside):
                inside += 1
            else:
                outside += 1
    counter +=1
    pbar.update(1)


    #cv2.imshow('Feed',frame1)
    #cv2.imshow('mask',mask)

    frame1 = frame2
    ret,frame2 = cap.read()


    key = cv2.waitKey(1)
    if key == 27:
        break
# Measure tracked path    
for i in range(len(pts)-1):
    dist = math.sqrt((pts[i+1][0] - pts[i][0])**2 + (pts[i+1][1] - pts[i][1])**2)
    total_path+=dist

if not os.path.exists('Results'):
    os.makedirs('Results')



if exp_type == 'plusmaze':
    df_plus = pd.DataFrame(columns=["video_name","right_side_time","left_side_time","up_side_time","down_side_time","mid_time","total_time_in_opened_area","total_time_in_closed_area","first_enterence","total_distance"])
    if os.path.isfile("Results/plusmaze.csv"):
        df_plus = pd.read_csv("Results/plusmaze.csv")
    result = [  
                os.path.splitext(os.path.basename(video_name))[0],
                round(right_/fps,2),
                round(left_/fps,2),
                round(up_/fps,2),
                round(down_/fps,2),
                round(mid_/fps,2),
                round((left_+right_)/fps,2),
                round((up_+down_)/fps,2),
                place[0],
                round((total_path*int(actual_area))/dist_exp_area,4)]
    df_plus.loc[len(df_plus.index)] = result
    df_plus.to_csv('Results/plusmaze.csv',index=False)
    print("Results are done!!!")

if exp_type == 'openfield':
    df_open = pd.DataFrame(columns=["video_name","Inside_time","outside_time","total_distance"])
    if os.path.isfile("Results/openfield.csv"):
        df_open = pd.read_csv("Results/openfield.csv")
    result = [                                
                os.path.splitext(os.path.basename(video_name))[0],
                round(inside/fps,2),
                round(outside/fps,2),
                round((total_path*int(actual_area))/dist_exp_area,4)]   
    df_open.loc[len(df_open.index)] = result
    df_open.to_csv('Results/openfield.csv',index=False)
    print("Results are done!!!")

cap.release()
cv2.destroyAllWindows()
