import cv2 as cv
import numpy as np


haar_cascade = cv.CascadeClassifier('ha_faceclas.xml')
artistes = ['Davido', 'Olamide', 'Tiwa Savage', 'Wizkid']
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
image =cv.imread(r'C:\Users\SHOPINVERSE\Desktop\cv_project\Test_Set\Davido\Screenshot_20230312-200113_1.jpg')
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("artiste", gray_image)
face_detect= haar_cascade.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 4)
for (x,y,w,h) in face_detect:
    faces_roi = gray_image[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print("label is {}".format(artistes[label]), "with a confidence level of {}".format(confidence))
    cv.putText(image, str(artistes[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 225, 255), thickness=2)
    cv.rectangle(image, (x,y), (x+w, y+h), (0, 225, 255), thickness=2)
cv.imshow("DETAILS OF THE ARTISTE RECOGNIZED", image)
cv.waitKey(0)
