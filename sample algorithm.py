import cv2
import numpy as np
face_cascade=cv2.CascadeClassifier('needle_4.xml')
cap=cv2.VideoCapture('traj_mot_1.mp4')


_, frame = cap.read()
rows, cols, _ = frame.shape
x_medium = int(cols / 2)
y_medium = int(rows / 2)
center = int(cols / 2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('r_1.mp4',fourcc, 5, (640,480))

while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h)in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img,'ram',(200,200),font,2,(255,255,0),2,cv2.LINE_AA)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
    frame1 = frame

    hsv_frame=cv2.cvtColor(frame1,cv2.COLOR_BGR2HSV)
    low_green=np.array([36,25,25])
    high_green=np.array([70,255,255])
    green_mask=cv2.inRange(hsv_frame,low_green,high_green)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)

    font=cv2.FONT_HERSHEY_SIMPLEX


    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        x_medium = int((x + x + w) / 2)
        y_medium = int((y + y + h) / 2)
        area=cv2.contourArea(cnt)

    
    cv2.imshow('Frame',frame1)

    b = cv2.resize(frame1,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    out.write(b)

    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()
